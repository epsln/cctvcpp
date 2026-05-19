#include "render_pipeline.h"
#include <algorithm>
#include <SDL2/SDL.h>

// ─────────────────────────────────────────────
//  Built-in blit vertex shader (shared by all passes)
// ─────────────────────────────────────────────
static const char* QUAD_VERT_SRC = R"GLSL(
#version 330 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aUV;
out vec2 vUV;
void main() { vUV = aUV; gl_Position = vec4(aPos, 0.0, 1.0); }
)GLSL";

// ─────────────────────────────────────────────
//  Passthrough blit (identity pass / screen output)
// ─────────────────────────────────────────────
static const char* BLIT_FRAG_SRC = R"GLSL(
#version 330 core
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uTex;
void main() { fragColor = texture(uTex, vUV); }
)GLSL";

// ─────────────────────────────────────────────
//  FBO
// ─────────────────────────────────────────────
bool FBO::create(int w, int h) {
    width = w; height = h;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
    bool ok = glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    if (!ok) SDL_Log("[FBO] Incomplete framebuffer");
    return ok;
}

void FBO::destroy() {
    if (fbo)     { glDeleteFramebuffers(1, &fbo);  fbo=0; }
    if (texture) { glDeleteTextures(1, &texture);  texture=0; }
}

// ─────────────────────────────────────────────
//  RenderPipeline
// ─────────────────────────────────────────────
RenderPipeline::~RenderPipeline() {
    m_fbo[0].destroy(); m_fbo[1].destroy();
    if (m_source_tex) glDeleteTextures(1, &m_source_tex);
    if (m_quad_vao)   glDeleteVertexArrays(1, &m_quad_vao);
    if (m_quad_vbo)   glDeleteBuffers(1, &m_quad_vbo);
    if (m_blit_program) glDeleteProgram(m_blit_program);
}

bool RenderPipeline::init(int width, int height, ShaderManager* sm) {
    m_width  = width;
    m_height = height;
    m_sm     = sm;

    // Source texture — allocated lazily at video resolution in render()
    // so it always matches the decoded frame dimensions exactly.
    // We start with a 1x1 placeholder so the texture object exists.
    glGenTextures(1, &m_source_tex);
    glBindTexture(GL_TEXTURE_2D, m_source_tex);
    uint8_t black[4] = {0,0,0,255};
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, black);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);
    m_source_tex_w = 1;
    m_source_tex_h = 1;

    // Ping-pong FBOs
    if (!m_fbo[0].create(width, height)) return false;
    if (!m_fbo[1].create(width, height)) return false;

    build_screen_quad();

    // Compile blit shader
    if (!sm->register_from_source("__blit__", QUAD_VERT_SRC, BLIT_FRAG_SRC)) return false;
    m_blit_program = sm->get_program("__blit__");

    SDL_Log("[RenderPipeline] Init OK %dx%d", width, height);
    return true;
}

void RenderPipeline::resize(int width, int height) {
    m_width = width; m_height = height;
    m_fbo[0].destroy(); m_fbo[1].destroy();
    m_fbo[0].create(width, height);
    m_fbo[1].create(width, height);
    // Source texture is video-sized — don't touch it here
}

void RenderPipeline::build_screen_quad() {
    // NDC full-screen quad (two triangles), UV flipped Y for GL convention
    float verts[] = {
        // X      Y     U     V
        -1.f, -1.f,  0.f, 0.f,
         1.f, -1.f,  1.f, 0.f,
         1.f,  1.f,  1.f, 1.f,
        -1.f, -1.f,  0.f, 0.f,
         1.f,  1.f,  1.f, 1.f,
        -1.f,  1.f,  0.f, 1.f,
    };
    glGenVertexArrays(1, &m_quad_vao);
    glGenBuffers(1, &m_quad_vbo);
    glBindVertexArray(m_quad_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_quad_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)(2*sizeof(float)));
    glBindVertexArray(0);
}

// ─────────────────────────────────────────────
//  Command queue (thread-safe)
// ─────────────────────────────────────────────
void RenderPipeline::push_command(EngineCommand cmd) {
    std::lock_guard<std::mutex> lk(m_cmd_mutex);
    m_cmd_queue.push_back(std::move(cmd));
}

void RenderPipeline::flush_commands() {
    std::deque<EngineCommand> cmds;
    { std::lock_guard<std::mutex> lk(m_cmd_mutex);
      std::swap(cmds, m_cmd_queue); }
    for (auto& cmd : cmds) {
        switch (cmd.type) {
            case CommandType::ADD_PASS:
                add_pass(cmd.pass_id, cmd.shader_name, cmd.position); break;
            case CommandType::REMOVE_PASS:
                remove_pass(cmd.pass_id); break;
            case CommandType::ENABLE_PASS:
                enable_pass(cmd.pass_id, cmd.enabled); break;
            case CommandType::SET_UNIFORM:
                set_uniform(cmd.pass_id, cmd.uniform_name, cmd.uniform_value); break;
            case CommandType::REORDER_PASS: {
                // move pass_id to cmd.position
                auto it = std::find_if(m_passes.begin(), m_passes.end(),
                    [&](const ShaderPass& p){ return p.id == cmd.pass_id; });
                if (it != m_passes.end()) {
                    ShaderPass p = std::move(*it);
                    m_passes.erase(it);
                    int idx = std::clamp(cmd.position, 0, (int)m_passes.size());
                    m_passes.insert(m_passes.begin()+idx, std::move(p));
                }
                break;
            }
            default: break;
        }
    }
}

// ─────────────────────────────────────────────
//  Direct pass management
// ─────────────────────────────────────────────
void RenderPipeline::add_pass(const std::string& id,
                               const std::string& shader_name,
                               int position) {
    GLuint prog = m_sm->get_program(shader_name);
    if (!prog) {
        SDL_Log("[RenderPipeline] Unknown shader: %s", shader_name.c_str());
        return;
    }
    // Remove if already exists
    remove_pass(id);

    ShaderPass pass;
    pass.id      = id;
    pass.name    = shader_name;
    pass.program = prog;
    pass.enabled = true;

    if (position < 0 || position >= (int)m_passes.size())
        m_passes.push_back(std::move(pass));
    else
        m_passes.insert(m_passes.begin() + position, std::move(pass));

    SDL_Log("[RenderPipeline] Added pass '%s' (shader=%s) at pos %d",
            id.c_str(), shader_name.c_str(), position);
}

void RenderPipeline::remove_pass(const std::string& id) {
    auto it = std::find_if(m_passes.begin(), m_passes.end(),
        [&](const ShaderPass& p){ return p.id == id; });
    if (it != m_passes.end()) {
        SDL_Log("[RenderPipeline] Removed pass '%s'", id.c_str());
        m_passes.erase(it);
    }
}

void RenderPipeline::enable_pass(const std::string& id, bool enabled) {
    for (auto& p : m_passes)
        if (p.id == id) { p.enabled = enabled; return; }
}

void RenderPipeline::set_uniform(const std::string& pass_id,
                                  const std::string& uniform_name,
                                  UniformValue val) {
    for (auto& p : m_passes)
        if (p.id == pass_id) { p.set_uniform(uniform_name, val); return; }
}

// ─────────────────────────────────────────────
//  Render one pass into the next FBO
// ─────────────────────────────────────────────
void RenderPipeline::render_pass(const ShaderPass& pass, GLuint input_tex,
                                  float time_sec, const CrowdState& crowd) {
    int dst = 1 - m_current_fbo;
    m_fbo[dst].bind();
    glViewport(0, 0, m_width, m_height);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(pass.program);

    // Bind input texture to unit 0
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, input_tex);
    GLint loc;
    if ((loc = glGetUniformLocation(pass.program, "uTex")) >= 0)
        glUniform1i(loc, 0);

    // Built-in uniforms (always available in all shaders)
    if ((loc = glGetUniformLocation(pass.program, "uTime"))      >= 0) glUniform1f(loc, time_sec);
    if ((loc = glGetUniformLocation(pass.program, "uResolution"))>= 0) glUniform2f(loc, (float)m_width, (float)m_height);
    if ((loc = glGetUniformLocation(pass.program, "uFrame"))     >= 0) glUniform1i(loc, m_frame_number);
    // Crowd uniforms
    if ((loc = glGetUniformLocation(pass.program, "uEnergy"))    >= 0) glUniform1f(loc, crowd.energy);
    if ((loc = glGetUniformLocation(pass.program, "uDensity"))   >= 0) glUniform1f(loc, crowd.density);
    if ((loc = glGetUniformLocation(pass.program, "uPulse"))     >= 0) glUniform1f(loc, crowd.pulse);
    if ((loc = glGetUniformLocation(pass.program, "uFrequency")) >= 0) glUniform1f(loc, crowd.frequency);
    if ((loc = glGetUniformLocation(pass.program, "uSentiment")) >= 0) glUniform1f(loc, crowd.sentiment);

    // Per-pass custom uniforms
    pass.apply_uniforms();

    glBindVertexArray(m_quad_vao);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);

    m_fbo[dst].unbind();
    m_current_fbo = dst;
}

// ─────────────────────────────────────────────
//  Blit final texture to the default framebuffer
// ─────────────────────────────────────────────
void RenderPipeline::blit_to_screen(GLuint tex) {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, m_width, m_height);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(m_blit_program);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);
    GLint loc = glGetUniformLocation(m_blit_program, "uTex");
    if (loc >= 0) glUniform1i(loc, 0);

    glBindVertexArray(m_quad_vao);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
}

// ─────────────────────────────────────────────
//  Main render tick
// ─────────────────────────────────────────────
RenderStats RenderPipeline::render(VideoSource& source,
                                    float time_sec,
                                    const CrowdState& crowd) {
    Uint64 t0 = SDL_GetPerformanceCounter();

    // 1. Drain command queue
    flush_commands();

    // 2. Reallocate source texture if video dimensions changed (first frame, or source switch)
    if (source.is_open()) {
        int vw = source.width(), vh = source.height();
        if (vw != m_source_tex_w || vh != m_source_tex_h) {
            glBindTexture(GL_TEXTURE_2D, m_source_tex);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, vw, vh, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            glBindTexture(GL_TEXTURE_2D, 0);
            m_source_tex_w = vw;
            m_source_tex_h = vh;
            SDL_Log("[RenderPipeline] Source texture resized to %dx%d", vw, vh);
        }
    }

    // 3. Upload new video frame into source texture (if available)
    source.upload_next_frame(m_source_tex);

    // 3. Run the pipeline
    //    Input to first pass is always the source texture.
    //    Each subsequent pass reads from the FBO the previous pass wrote to.

    GLuint current_input = m_source_tex;

    // Reset FBO to a known starting slot
    m_current_fbo = 0;

    // If no passes: just blit source directly
    if (m_passes.empty()) {
        blit_to_screen(m_source_tex);
    } else {
        int active_count = 0;
        for (auto& pass : m_passes) {
            if (!pass.enabled) continue;
            render_pass(pass, current_input, time_sec, crowd);
            current_input = m_fbo[m_current_fbo].texture;
            ++active_count;
        }
        blit_to_screen(current_input);
    }

    ++m_frame_number;

    Uint64 t1 = SDL_GetPerformanceCounter();
    double freq = (double)SDL_GetPerformanceFrequency();

    RenderStats stats;
    stats.frame_time_ms = (double)(t1-t0) / freq * 1000.0;
    stats.active_passes = (int)std::count_if(m_passes.begin(), m_passes.end(),
        [](const ShaderPass& p){ return p.enabled; });
    stats.frame_number  = m_frame_number;
    return stats;
}
