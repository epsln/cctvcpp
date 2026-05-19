#include "shader_manager.h"
#include <fstream>
#include <sstream>
#include <SDL2/SDL.h>

ShaderManager::~ShaderManager() {
    for (auto& [name, prog] : m_programs)
        if (prog) glDeleteProgram(prog);
}

std::string ShaderManager::read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) { SDL_Log("[ShaderManager] Cannot open: %s", path.c_str()); return ""; }
    std::ostringstream ss; ss << f.rdbuf();
    return ss.str();
}

GLuint ShaderManager::compile_shader(GLenum type, const std::string& src) const {
    GLuint sh = glCreateShader(type);
    const char* cstr = src.c_str();
    glShaderSource(sh, 1, &cstr, nullptr);
    glCompileShader(sh);
    GLint ok; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024]; glGetShaderInfoLog(sh, sizeof(log), nullptr, log);
        SDL_Log("[ShaderManager] Compile error (%s):\n%s",
                type==GL_VERTEX_SHADER?"VERT":"FRAG", log);
        glDeleteShader(sh);
        return 0;
    }
    return sh;
}

GLuint ShaderManager::link_program(GLuint vert, GLuint frag) const {
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vert);
    glAttachShader(prog, frag);
    glLinkProgram(prog);
    GLint ok; glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024]; glGetProgramInfoLog(prog, sizeof(log), nullptr, log);
        SDL_Log("[ShaderManager] Link error:\n%s", log);
        glDeleteProgram(prog);
        return 0;
    }
    return prog;
}

bool ShaderManager::register_from_source(const std::string& name,
                                          const std::string& vert_src,
                                          const std::string& frag_src) {
    GLuint vert = compile_shader(GL_VERTEX_SHADER,   vert_src);
    GLuint frag = compile_shader(GL_FRAGMENT_SHADER, frag_src);
    if (!vert || !frag) { glDeleteShader(vert); glDeleteShader(frag); return false; }
    GLuint prog = link_program(vert, frag);
    glDeleteShader(vert); glDeleteShader(frag);
    if (!prog) return false;
    if (m_programs.count(name)) glDeleteProgram(m_programs[name]);
    m_programs[name] = prog;
    SDL_Log("[ShaderManager] Registered shader: %s", name.c_str());
    return true;
}

bool ShaderManager::register_from_files(const std::string& name,
                                         const std::string& vert_path,
                                         const std::string& frag_path) {
    std::string vs = read_file(vert_path);
    std::string fs = read_file(frag_path);
    if (vs.empty() || fs.empty()) return false;
    return register_from_source(name, vs, fs);
}

GLuint ShaderManager::get_program(const std::string& name) const {
    auto it = m_programs.find(name);
    return it != m_programs.end() ? it->second : 0;
}

std::vector<std::string> ShaderManager::names() const {
    std::vector<std::string> out;
    for (auto& [k,v] : m_programs) out.push_back(k);
    return out;
}
