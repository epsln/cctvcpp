#include "vj_engine.h"
#include "video_source.h"
#include "shader_manager.h"
#include "render_pipeline.h"

#include <SDL2/SDL.h>
#include <GL/glew.h>
#include <SDL2/SDL_opengl.h>

#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <chrono>

// ─────────────────────────────────────────────
//  Config
// ─────────────────────────────────────────────
struct Config {
    int   width      = 1280;
    int   height     = 720;
    int   target_fps = 60;
    std::string video_path;
    std::string shader_dir = "shaders/";
};

// ─────────────────────────────────────────────
//  Simulated crowd state (placeholder for RL)
//  In the real system this is filled by the RL agent.
// ─────────────────────────────────────────────
static CrowdState sim_crowd(float t) {
    CrowdState cs;
    cs.energy    = 0.5f + 0.4f * std::sin(t * 0.3f);
    cs.density   = 0.5f + 0.3f * std::sin(t * 0.17f + 1.f);
    cs.pulse     = (std::fmod(t, 0.5f) < 0.05f) ? 1.f : 0.f;  // 120 BPM kick
    cs.frequency = 0.5f + 0.4f * std::sin(t * 2.1f);
    cs.sentiment = std::sin(t * 0.1f);
    return cs;
}

// ─────────────────────────────────────────────
//  Print usage
// ─────────────────────────────────────────────
static void print_help() {
    SDL_Log("=== VJ Engine Controls ===");
    SDL_Log("  1 - Toggle chromatic aberration");
    SDL_Log("  2 - Toggle glitch");
    SDL_Log("  3 - Toggle color grade");
    SDL_Log("  4 - Toggle bloom");
    SDL_Log("  5 - Toggle kaleidoscope");
    SDL_Log("  A - Add glitch pass  (demo RL ADD_PASS command)");
    SDL_Log("  R - Remove glitch pass");
    SDL_Log("  E - Boost energy (demo SET_UNIFORM)");
    SDL_Log("  F - Toggle fullscreen");
    SDL_Log("  Q/Esc - Quit");
}

// ─────────────────────────────────────────────
//  Entry point
// ─────────────────────────────────────────────
int main(int argc, char* argv[]) {
    Config cfg;
    if (argc < 2) {
        fprintf(stderr, "Usage: vjengine <video_file> [width] [height]\n");
        fprintf(stderr, "       A test pattern will be used if the file cannot be opened.\n");
    } else {
        cfg.video_path = argv[1];
    }
    if (argc >= 4) {
        cfg.width  = std::atoi(argv[2]);
        cfg.height = std::atoi(argv[3]);
    }

    // ─── SDL / OpenGL init ────────────────────
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) < 0) {
        fprintf(stderr, "SDL_Init: %s\n", SDL_GetError());
        return 1;
    }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

    SDL_Window* window = SDL_CreateWindow(
        "VJ Engine",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        cfg.width, cfg.height,
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
    if (!window) { fprintf(stderr, "SDL_CreateWindow: %s\n", SDL_GetError()); return 1; }

    SDL_GLContext gl_ctx = SDL_GL_CreateContext(window);
    if (!gl_ctx) { fprintf(stderr, "SDL_GL_CreateContext: %s\n", SDL_GetError()); return 1; }

    SDL_GL_SetSwapInterval(1); // vsync

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) { fprintf(stderr, "glewInit failed\n"); return 1; }

    SDL_Log("GL: %s / GLSL: %s",
        (const char*)glGetString(GL_VERSION),
        (const char*)glGetString(GL_SHADING_LANGUAGE_VERSION));

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glClearColor(0.f, 0.f, 0.f, 1.f);

    // ─── Shader manager ──────────────────────
    ShaderManager sm;

    // Load all built-in shaders (quad.vert + each .frag)
    std::string vpath = cfg.shader_dir + "quad.vert";
    struct ShaderDef { const char* name; const char* frag; };
    std::vector<ShaderDef> defs = {
        {"chromatic",     "shaders/chromatic.frag"},
        {"glitch",        "shaders/glitch.frag"},
        {"color_grade",   "shaders/color_grade.frag"},
        {"bloom",         "shaders/bloom.frag"},
        {"kaleidoscope",  "shaders/kaleidoscope.frag"},
        {"feedback",      "shaders/feedback.frag"},
    };
    for (auto& d : defs) {
        if (!sm.register_from_files(d.name, vpath, d.frag))
            SDL_Log("[main] WARNING: Could not load shader: %s", d.name);
    }

    // ─── Render pipeline ─────────────────────
    RenderPipeline pipeline;
    if (!pipeline.init(cfg.width, cfg.height, &sm)) {
        fprintf(stderr, "pipeline.init failed\n");
        return 1;
    }

    // Default passes at startup
    pipeline.add_pass("chroma",   "chromatic",   -1);
    pipeline.add_pass("grade",    "color_grade", -1);
    pipeline.add_pass("bloom_fx", "bloom",       -1);

    // Set some default uniforms
    pipeline.set_uniform("chroma",   "uStrength",    UniformValue::from_float(0.008f));
    pipeline.set_uniform("chroma",   "uBarrel",      UniformValue::from_float(0.1f));
    pipeline.set_uniform("grade",    "uSaturation",  UniformValue::from_float(1.3f));
    pipeline.set_uniform("grade",    "uContrast",    UniformValue::from_float(1.1f));
    pipeline.set_uniform("grade",    "uHueShift",    UniformValue::from_float(5.0f));
    pipeline.set_uniform("bloom_fx", "uThreshold",   UniformValue::from_float(0.55f));
    pipeline.set_uniform("bloom_fx", "uIntensity",   UniformValue::from_float(1.2f));
    pipeline.set_uniform("bloom_fx", "uRadius",      UniformValue::from_int(6));

    // ─── Video source ─────────────────────────
    VideoSource source;
    if (!cfg.video_path.empty())
        source.open(cfg.video_path);
    else
        SDL_Log("[main] No video file provided; rendering black source.");

    // ─── Timing ──────────────────────────────
    Uint64 perf_freq  = SDL_GetPerformanceFrequency();
    Uint64 last_tick  = SDL_GetPerformanceCounter();
    float  time_sec   = 0.f;
    bool   fullscreen = false;

    print_help();

    // ─── Main loop ───────────────────────────
    bool running = true;
    while (running) {
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_QUIT) { running = false; break; }
            if (ev.type == SDL_WINDOWEVENT &&
                ev.window.event == SDL_WINDOWEVENT_RESIZED) {
                cfg.width  = ev.window.data1;
                cfg.height = ev.window.data2;
                pipeline.resize(cfg.width, cfg.height);
            }
            if (ev.type == SDL_KEYDOWN) {
                auto& k = ev.key.keysym.sym;
                if (k == SDLK_ESCAPE || k == SDLK_q) { running = false; }
                if (k == SDLK_f) {
                    fullscreen = !fullscreen;
                    SDL_SetWindowFullscreen(window,
                        fullscreen ? SDL_WINDOW_FULLSCREEN_DESKTOP : 0);
                }
                // Toggle passes by keyboard (simulating RL enable/disable commands)
                if (k == SDLK_1) {
                    EngineCommand cmd; cmd.type=CommandType::ENABLE_PASS;
                    cmd.pass_id="chroma";
                    // Find current state (simple toggle by re-checking)
                    static bool chroma_on=true; chroma_on=!chroma_on;
                    cmd.enabled=chroma_on;
                    pipeline.push_command(cmd);
                    SDL_Log("Chromatic: %s", chroma_on?"ON":"OFF");
                }
                if (k == SDLK_2) {
                    static bool glitch_on=false; glitch_on=!glitch_on;
                    EngineCommand cmd; cmd.type=CommandType::ENABLE_PASS;
                    cmd.pass_id="glitch_fx"; cmd.enabled=glitch_on;
                    pipeline.push_command(cmd);
                    SDL_Log("Glitch: %s", glitch_on?"ON":"OFF");
                }
                if (k == SDLK_3) {
                    static bool grade_on=true; grade_on=!grade_on;
                    EngineCommand cmd; cmd.type=CommandType::ENABLE_PASS;
                    cmd.pass_id="grade"; cmd.enabled=grade_on;
                    pipeline.push_command(cmd);
                    SDL_Log("Color grade: %s", grade_on?"ON":"OFF");
                }
                if (k == SDLK_4) {
                    static bool bloom_on=true; bloom_on=!bloom_on;
                    EngineCommand cmd; cmd.type=CommandType::ENABLE_PASS;
                    cmd.pass_id="bloom_fx"; cmd.enabled=bloom_on;
                    pipeline.push_command(cmd);
                    SDL_Log("Bloom: %s", bloom_on?"ON":"OFF");
                }
                if (k == SDLK_5) {
                    static bool kaleido_on=false; kaleido_on=!kaleido_on;
                    if (kaleido_on) {
                        EngineCommand cmd; cmd.type=CommandType::ADD_PASS;
                        cmd.pass_id="kaleido"; cmd.shader_name="kaleidoscope"; cmd.position=0;
                        pipeline.push_command(cmd);
                        EngineCommand ucmd; ucmd.type=CommandType::SET_UNIFORM;
                        ucmd.pass_id="kaleido"; ucmd.uniform_name="uSegments";
                        ucmd.uniform_value=UniformValue::from_float(6.f);
                        pipeline.push_command(ucmd);
                    } else {
                        EngineCommand cmd; cmd.type=CommandType::REMOVE_PASS;
                        cmd.pass_id="kaleido";
                        pipeline.push_command(cmd);
                    }
                    SDL_Log("Kaleidoscope: %s", kaleido_on?"ON":"OFF");
                }
                // Demo RL command: ADD_PASS
                if (k == SDLK_a) {
                    EngineCommand cmd; cmd.type=CommandType::ADD_PASS;
                    cmd.pass_id="glitch_fx"; cmd.shader_name="glitch"; cmd.position=1;
                    pipeline.push_command(cmd);
                    EngineCommand ucmd; ucmd.type=CommandType::SET_UNIFORM;
                    ucmd.pass_id="glitch_fx"; ucmd.uniform_name="uAmount";
                    ucmd.uniform_value=UniformValue::from_float(0.04f);
                    pipeline.push_command(ucmd);
                    SDL_Log("[RL] Added glitch pass");
                }
                // Demo RL command: REMOVE_PASS
                if (k == SDLK_r) {
                    EngineCommand cmd; cmd.type=CommandType::REMOVE_PASS;
                    cmd.pass_id="glitch_fx";
                    pipeline.push_command(cmd);
                    SDL_Log("[RL] Removed glitch pass");
                }
                // Demo RL command: SET_UNIFORM
                if (k == SDLK_e) {
                    static float energy_boost = 0.f;
                    energy_boost = (energy_boost > 0.5f) ? 0.f : 1.0f;
                    EngineCommand cmd; cmd.type=CommandType::SET_UNIFORM;
                    cmd.pass_id="chroma"; cmd.uniform_name="uStrength";
                    cmd.uniform_value=UniformValue::from_float(energy_boost > 0.5f ? 0.03f : 0.008f);
                    pipeline.push_command(cmd);
                    SDL_Log("[RL] Energy boost: %.1f", energy_boost);
                }
            }
        }

        // Advance time
        Uint64 now  = SDL_GetPerformanceCounter();
        float  dt   = (float)(now - last_tick) / (float)perf_freq;
        last_tick   = now;
        time_sec   += dt;

        // Simulated crowd state (replace with real RL output)
        CrowdState crowd = sim_crowd(time_sec);

        // Render
        RenderStats stats = pipeline.render(source, time_sec, crowd);

        SDL_GL_SwapWindow(window);

        // Print stats every 5 seconds
        if ((int)(time_sec) % 5 == 0 && dt < 0.02f) {
            SDL_Log("[Stats] frame=%d  passes=%d  frame_ms=%.2f  energy=%.2f",
                stats.frame_number, stats.active_passes,
                stats.frame_time_ms, crowd.energy);
        }
    }

    source.close();
    SDL_GL_DeleteContext(gl_ctx);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
