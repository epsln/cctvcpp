#pragma once

#include <SDL2/SDL.h>
#include <GL/glew.h>
#include <SDL2/SDL_opengl.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <unordered_map>
#include <deque>
#include <mutex>
#include <atomic>

// ─────────────────────────────────────────────
//  Forward declarations
// ─────────────────────────────────────────────
struct ShaderPass;
struct Pipeline;
class VideoSource;
class ShaderManager;
class RenderPipeline;
class VJEngine;

// ─────────────────────────────────────────────
//  Shader uniform value (int, float, vec2/3/4)
// ─────────────────────────────────────────────
struct UniformValue {
    enum class Type { INT, FLOAT, VEC2, VEC3, VEC4 } type;
    union {
        int   i;
        float f;
        float v[4];
    };
    static UniformValue from_int(int v)            { UniformValue u; u.type=Type::INT;   u.i=v;                               return u; }
    static UniformValue from_float(float v)        { UniformValue u; u.type=Type::FLOAT; u.f=v;                               return u; }
    static UniformValue from_vec2(float x,float y) { UniformValue u; u.type=Type::VEC2;  u.v[0]=x; u.v[1]=y;                  return u; }
    static UniformValue from_vec3(float x,float y,float z)        { UniformValue u; u.type=Type::VEC3; u.v[0]=x;u.v[1]=y;u.v[2]=z; return u; }
    static UniformValue from_vec4(float x,float y,float z,float w){ UniformValue u; u.type=Type::VEC4; u.v[0]=x;u.v[1]=y;u.v[2]=z;u.v[3]=w; return u; }
};

// ─────────────────────────────────────────────
//  A single shader pass in the pipeline
// ─────────────────────────────────────────────
struct ShaderPass {
    std::string id;           // unique id used by RL commands
    std::string name;         // human-readable
    GLuint      program = 0;
    bool        enabled = true;
    std::unordered_map<std::string, UniformValue> uniforms;

    void set_uniform(const std::string& name_u, UniformValue val) { uniforms[name_u] = val; }
    void apply_uniforms() const;
};

// ─────────────────────────────────────────────
//  Command interface (used by the RL engine)
// ─────────────────────────────────────────────
enum class CommandType {
    ADD_PASS,       // add a shader pass at position
    REMOVE_PASS,    // remove pass by id
    ENABLE_PASS,    // toggle on/off
    SET_UNIFORM,    // set a uniform on a pass
    SET_SOURCE,     // switch video/image source
    SET_BLEND,      // change blending params
    REORDER_PASS,   // move pass to index
};

struct EngineCommand {
    CommandType type;
    std::string pass_id;
    std::string shader_name;    // for ADD_PASS
    int         position = -1;  // for ADD_PASS / REORDER_PASS (-1 = append)
    bool        enabled  = true; // for ENABLE_PASS
    std::string uniform_name;
    UniformValue uniform_value = UniformValue::from_float(0.f);
    std::string source_path;    // for SET_SOURCE
};

// ─────────────────────────────────────────────
//  Crowd state (fed by external sensor / RL)
// ─────────────────────────────────────────────
struct CrowdState {
    float energy      = 0.f;   // 0-1
    float density     = 0.f;   // 0-1
    float pulse       = 0.f;   // beat strength
    float frequency   = 0.f;   // dominant audio freq normalised
    float sentiment   = 0.f;   // -1 calm .. +1 excited
};

// ─────────────────────────────────────────────
//  Render stats (returned to RL engine)
// ─────────────────────────────────────────────
struct RenderStats {
    double frame_time_ms  = 0.0;
    int    active_passes  = 0;
    int    frame_number   = 0;
};
