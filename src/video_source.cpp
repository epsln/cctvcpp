#include "video_source.h"
#include <cstdio>
#include <cstring>

static double wall_time_sec() {
    return SDL_GetTicks64() / 1000.0;
}

VideoSource::VideoSource() = default;

VideoSource::~VideoSource() {
    close();
}

bool VideoSource::open(const std::string& path) {
    close();
    m_path = path;

    // Open container
    if (avformat_open_input(&m_fmt_ctx, path.c_str(), nullptr, nullptr) < 0) {
        SDL_Log("[VideoSource] Cannot open: %s", path.c_str());
        return false;
    }
    if (avformat_find_stream_info(m_fmt_ctx, nullptr) < 0) {
        SDL_Log("[VideoSource] Cannot find stream info");
        return false;
    }

    // Find first video stream
    m_video_stream = -1;
    for (unsigned i = 0; i < m_fmt_ctx->nb_streams; ++i) {
        if (m_fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            m_video_stream = (int)i;
            break;
        }
    }
    if (m_video_stream < 0) {
        SDL_Log("[VideoSource] No video stream found");
        return false;
    }

    AVStream* vs = m_fmt_ctx->streams[m_video_stream];
    m_time_base = av_q2d(vs->time_base);
    m_width  = vs->codecpar->width;
    m_height = vs->codecpar->height;

    // Open codec
    const AVCodec* codec = avcodec_find_decoder(vs->codecpar->codec_id);
    if (!codec) { SDL_Log("[VideoSource] No decoder"); return false; }
    m_codec_ctx = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(m_codec_ctx, vs->codecpar);
    m_codec_ctx->thread_count = 2;
    if (avcodec_open2(m_codec_ctx, codec, nullptr) < 0) {
        SDL_Log("[VideoSource] avcodec_open2 failed");
        return false;
    }

    // SwsContext: decode to RGBA
    m_sws_ctx = sws_getContext(
        m_width, m_height, m_codec_ctx->pix_fmt,
        m_width, m_height, AV_PIX_FMT_RGBA,
        SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (!m_sws_ctx) { SDL_Log("[VideoSource] sws_getContext failed"); return false; }

    m_open    = true;
    m_eof     = false;
    m_running = true;
    m_playback_start_wall = wall_time_sec();
    m_last_pts      = 0.0;
    m_loop_pts_base = 0.0;   // accumulated PTS offset across loops

    m_worker = std::thread(&VideoSource::decode_loop, this);
    SDL_Log("[VideoSource] Opened %s (%dx%d)", path.c_str(), m_width, m_height);
    return true;
}

void VideoSource::close() {
    if (m_running) {
        m_running = false;
        m_queue_cv.notify_all();
        if (m_worker.joinable()) m_worker.join();
    }
    { std::lock_guard<std::mutex> lk(m_queue_mutex);
      while (!m_frame_queue.empty()) m_frame_queue.pop(); }
    if (m_sws_ctx)   { sws_freeContext(m_sws_ctx);      m_sws_ctx   = nullptr; }
    if (m_codec_ctx) { avcodec_free_context(&m_codec_ctx); }
    if (m_fmt_ctx)   { avformat_close_input(&m_fmt_ctx); }
    m_open = false;
    m_eof  = false;
}

// ─────────────────────────────────────────────
//  Worker thread: decode frames into queue
// ─────────────────────────────────────────────
void VideoSource::decode_loop() {
    AVPacket* pkt   = av_packet_alloc();
    AVFrame*  frame = av_frame_alloc();

    // Local loop state — never touch m_playback_start_wall from this thread
    double loop_pts_base  = 0.0;   // monotonic PTS offset accumulated across loops
    double last_local_pts = 0.0;   // last raw PTS seen in this loop pass
    double loop_duration  = 0.0;   // measured duration of the last complete loop

    auto push_frame = [&](AVFrame* f) {
        double raw_pts = (f->pts != AV_NOPTS_VALUE)
            ? f->pts * m_time_base
            : last_local_pts + 1.0 / 25.0;

        // Detect PTS going backwards (can happen on seek) — ignore stale frames
        if (raw_pts < last_local_pts - 0.5) return;

        last_local_pts = raw_pts;
        loop_duration  = raw_pts;   // track max PTS seen this pass

        DecodedFrame df;
        df.width   = m_width;
        df.height  = m_height;
        df.pts_sec = loop_pts_base + raw_pts;   // monotonic

        df.data.resize((size_t)m_width * m_height * 4);
        uint8_t* dst_data[1] = { df.data.data() };
        int      dst_line[1] = { m_width * 4 };
        sws_scale(m_sws_ctx, f->data, f->linesize, 0, m_height, dst_data, dst_line);

        std::unique_lock<std::mutex> lk(m_queue_mutex);
        m_queue_cv.wait(lk, [&]{ return !m_running || (int)m_frame_queue.size() < QUEUE_SIZE; });
        if (m_running) m_frame_queue.push(std::move(df));
        lk.unlock();
        m_queue_cv.notify_one();
    };

    while (m_running) {
        int ret = av_read_frame(m_fmt_ctx, pkt);
        if (ret < 0) {
            // EOF — advance the monotonic base by the measured duration of this pass
            // (add one frame duration to avoid the last frame being re-shown)
            double frame_dur = (m_codec_ctx->framerate.num > 0)
                ? (double)m_codec_ctx->framerate.den / m_codec_ctx->framerate.num
                : 1.0 / 25.0;
            loop_pts_base  += loop_duration + frame_dur;
            last_local_pts  = 0.0;
            loop_duration   = 0.0;

            av_seek_frame(m_fmt_ctx, m_video_stream, 0, AVSEEK_FLAG_BACKWARD);
            avcodec_flush_buffers(m_codec_ctx);
            continue;
        }
        if (pkt->stream_index == m_video_stream) {
            avcodec_send_packet(m_codec_ctx, pkt);
            while (avcodec_receive_frame(m_codec_ctx, frame) == 0)
                push_frame(frame);
        }
        av_packet_unref(pkt);
    }

    av_frame_free(&frame);
    av_packet_free(&pkt);
}

// ─────────────────────────────────────────────
//  Called from GL thread: upload if a new frame is due
// ─────────────────────────────────────────────
bool VideoSource::upload_next_frame(GLuint texture_id) {
    double now = wall_time_sec() - m_playback_start_wall;

    std::unique_lock<std::mutex> lk(m_queue_mutex);
    if (m_frame_queue.empty()) return false;

    // Skip stale frames: if the next frame is already late, keep popping
    // until we find the freshest frame that is still due (or the last due one).
    // This prevents a burst of frame uploads after a hitch.
    bool uploaded = false;
    while (!m_frame_queue.empty()) {
        const DecodedFrame& front = m_frame_queue.front();

        // Frame not due yet — stop
        if (front.pts_sec > now + 0.001) break;

        // Check if the *next* frame is also overdue; if so, skip this one
        // (we only upload the most recent due frame)
        bool next_is_also_due = (m_frame_queue.size() >= 2);
        if (next_is_also_due) {
            // Peek at the second element
            // std::queue doesn't expose iterators; copy-pop to check
            DecodedFrame skipped = std::move(m_frame_queue.front());
            m_frame_queue.pop();
            // Check front again
            if (!m_frame_queue.empty() && m_frame_queue.front().pts_sec <= now + 0.001) {
                // The next frame is also due — discard skipped, continue draining
                uploaded = true; // mark that at least one frame was consumed
                continue;
            } else {
                // Next frame is not due — restore skipped (put it back isn't possible,
                // so just upload it now since it's the right one)
                lk.unlock();
                m_queue_cv.notify_one();
                glBindTexture(GL_TEXTURE_2D, texture_id);
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, skipped.width, skipped.height,
                                GL_RGBA, GL_UNSIGNED_BYTE, skipped.data.data());
                glBindTexture(GL_TEXTURE_2D, 0);
                return true;
            }
        }

        // This is the right frame — upload it
        DecodedFrame df = std::move(m_frame_queue.front());
        m_frame_queue.pop();
        lk.unlock();
        m_queue_cv.notify_one();

        glBindTexture(GL_TEXTURE_2D, texture_id);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, df.width, df.height,
                        GL_RGBA, GL_UNSIGNED_BYTE, df.data.data());
        glBindTexture(GL_TEXTURE_2D, 0);
        return true;
    }

    if (uploaded) m_queue_cv.notify_one();
    return uploaded;
}
