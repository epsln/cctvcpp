#pragma once
#include "vj_engine.h"
#include <string>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

// ─────────────────────────────────────────────
//  Decoded frame (RGBA, GPU-ready)
// ─────────────────────────────────────────────
struct DecodedFrame {
    std::vector<uint8_t> data;
    int width = 0, height = 0;
    double pts_sec = 0.0;
};

// ─────────────────────────────────────────────
//  VideoSource
//  Owns an FFmpeg decode loop on a worker thread.
//  Exposes upload_next_frame() to push the
//  current frame into a GL texture (call from GL thread).
// ─────────────────────────────────────────────
class VideoSource {
public:
    VideoSource();
    ~VideoSource();

    bool open(const std::string& path);
    void close();

    // Called from the GL thread each render tick.
    // Returns true if the texture was updated.
    bool upload_next_frame(GLuint texture_id);

    int width()  const { return m_width;  }
    int height() const { return m_height; }
    bool is_open() const { return m_open; }
    const std::string& path() const { return m_path; }

private:
    void decode_loop();

    std::string m_path;
    int m_width  = 0;
    int m_height = 0;
    bool m_open  = false;

    // FFmpeg handles
    AVFormatContext* m_fmt_ctx   = nullptr;
    AVCodecContext*  m_codec_ctx = nullptr;
    SwsContext*      m_sws_ctx   = nullptr;
    int              m_video_stream = -1;
    double           m_time_base   = 1.0;

    // Frame queue (decoded on worker, consumed on GL thread)
    static constexpr int QUEUE_SIZE = 4;
    std::mutex              m_queue_mutex;
    std::condition_variable m_queue_cv;
    std::queue<DecodedFrame> m_frame_queue;
    bool m_eof = false;

    // Worker thread
    std::thread      m_worker;
    std::atomic<bool> m_running{false};

    // Timing
    double m_playback_start_wall = 0.0;  // SDL_GetTicks64 at open (GL thread only — never written by worker)
    double m_last_pts = 0.0;
    double m_loop_pts_base = 0.0;        // unused externally; bookkeeping in decode_loop
};
