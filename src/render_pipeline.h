#pragma once
#include "vj_engine.h"
#include "video_source.h"
#include "shader_manager.h"
#include <vector>
#include <memory>
#include <deque>
#include <mutex>

// ─────────────────────────────────────────────
//  Framebuffer Object wrapper
// ─────────────────────────────────────────────
struct FBO {
    GLuint fbo     = 0;
    GLuint texture = 0;
    int    width   = 0;
    int    height  = 0;

    bool create(int w, int h);
    void destroy();
    void bind()   const { glBindFramebuffer(GL_FRAMEBUFFER, fbo); }
    void unbind() const { glBindFramebuffer(GL_FRAMEBUFFER, 0); }
};

// ─────────────────────────────────────────────
//  RenderPipeline
//  source texture → pass[0] → pass[1] → ... → screen
//
//  Ping-pong between two FBOs.
//  Each pass reads from one FBO and writes to the other.
// ─────────────────────────────────────────────
class RenderPipeline {
public:
    RenderPipeline() = default;
    ~RenderPipeline();

    bool init(int width, int height, ShaderManager* sm);
    void resize(int width, int height);

    // Called each frame:
    // 1. upload video frame into source_tex if new frame available
    // 2. run all enabled passes
    // 3. blit final texture to screen quad
    RenderStats render(VideoSource& source, float time_sec, const CrowdState& crowd);

    // ─── Command interface (thread-safe) ───────
    void push_command(EngineCommand cmd);
    void flush_commands();   // called from GL thread before render

    // Direct accessors (for setup from main thread before render loop)
    void add_pass(const std::string& id, const std::string& shader_name, int position = -1);
    void remove_pass(const std::string& id);
    void enable_pass(const std::string& id, bool enabled);
    void set_uniform(const std::string& pass_id,
                     const std::string& uniform_name,
                     UniformValue val);

    int pass_count() const { return (int)m_passes.size(); }

private:
    void build_screen_quad();
    void render_pass(const ShaderPass& pass, GLuint input_tex,
                     float time_sec, const CrowdState& crowd);
    void blit_to_screen(GLuint tex);

    int m_width = 0, m_height = 0;
    ShaderManager* m_sm = nullptr;

    // Source frame texture (uploaded from video, sized to VIDEO dims not window)
    GLuint m_source_tex   = 0;
    int    m_source_tex_w = 0;
    int    m_source_tex_h = 0;

    // Ping-pong FBOs
    FBO m_fbo[2];
    int m_current_fbo = 0;  // index of the FBO we just wrote to

    // Quad VAO/VBO for full-screen passes
    GLuint m_quad_vao = 0;
    GLuint m_quad_vbo = 0;

    // Pass list
    std::vector<ShaderPass> m_passes;

    // Screen blit shader
    GLuint m_blit_program = 0;

    // Command queue (filled by RL engine, drained on GL thread)
    std::deque<EngineCommand> m_cmd_queue;
    std::mutex                m_cmd_mutex;

    int m_frame_number = 0;
};
