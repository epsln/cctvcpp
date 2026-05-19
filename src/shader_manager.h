#pragma once
#include "vj_engine.h"
#include <string>
#include <unordered_map>

// ─────────────────────────────────────────────
//  ShaderManager
//  Loads GLSL from files or inline strings,
//  compiles, links, and caches programs.
// ─────────────────────────────────────────────
class ShaderManager {
public:
    ShaderManager() = default;
    ~ShaderManager();

    // Register a shader from GLSL source strings
    bool register_from_source(const std::string& name,
                               const std::string& vert_src,
                               const std::string& frag_src);

    // Register from .vert / .frag files
    bool register_from_files(const std::string& name,
                              const std::string& vert_path,
                              const std::string& frag_path);

    // Get a compiled program (0 = not found)
    GLuint get_program(const std::string& name) const;

    // List registered shader names
    std::vector<std::string> names() const;

private:
    GLuint compile_shader(GLenum type, const std::string& src) const;
    GLuint link_program(GLuint vert, GLuint frag) const;
    static std::string read_file(const std::string& path);

    std::unordered_map<std::string, GLuint> m_programs;
};

// ─────────────────────────────────────────────
//  ShaderPass uniform application
// ─────────────────────────────────────────────
inline void ShaderPass::apply_uniforms() const {
    for (auto& [name_u, val] : uniforms) {
        GLint loc = glGetUniformLocation(program, name_u.c_str());
        if (loc < 0) continue;
        switch (val.type) {
            case UniformValue::Type::INT:   glUniform1i(loc, val.i);  break;
            case UniformValue::Type::FLOAT: glUniform1f(loc, val.f);  break;
            case UniformValue::Type::VEC2:  glUniform2fv(loc,1,val.v); break;
            case UniformValue::Type::VEC3:  glUniform3fv(loc,1,val.v); break;
            case UniformValue::Type::VEC4:  glUniform4fv(loc,1,val.v); break;
        }
    }
}
