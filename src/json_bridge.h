#pragma once
/*
 * json_bridge.h
 * ─────────────
 * Minimal JSON read/write for the VJ engine ↔ RL agent file protocol.
 *
 * No external JSON library required — we only need a small subset:
 *   Write: EngineState → JSON string → vj_state.json  (atomic via tmp rename)
 *   Read:  vj_commands.json → list of EngineCommands
 *
 * Format is strict but readable. Numbers are printed with 6 decimal places.
 * Strings are escaped only for the characters we actually produce/consume
 * (file paths on Linux — no control chars, no backslashes outside of paths).
 *
 * Usage (from main loop):
 *   JsonBridge bridge("vj_state.json", "vj_commands.json");
 *   // each frame:
 *   bridge.write_state(state);
 *   auto cmds = bridge.read_commands();
 *   for (auto& cmd : cmds) pipeline.push_command(cmd);
 */

#include "vj_engine.h"
#include "video_source.h"
#include "render_pipeline.h"

#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cassert>

extern "C" {
#include <libavformat/avformat.h>
}

// ─────────────────────────────────────────────
//  Extended engine state for the bridge
// ─────────────────────────────────────────────
struct BridgeState {
    RenderStats   render;
    CrowdState    crowd;
    bool          has_source      = false;
    std::string   source_path;
    double        source_pos_sec  = 0.0;
    double        source_dur_sec  = 0.0;
    bool          source_near_end = false;
    // Pass list: id, shader, enabled
    struct PassInfo { std::string id, shader; bool enabled; };
    std::vector<PassInfo> passes;
    double        wall_time       = 0.0;
};

// ─────────────────────────────────────────────
//  Tiny JSON helpers
// ─────────────────────────────────────────────
namespace jw {

static std::string esc(const std::string& s) {
    std::string out; out.reserve(s.size() + 4);
    for (char c : s) {
        if (c == '"')  { out += "\\\""; }
        else if (c == '\\') { out += "\\\\"; }
        else           { out += c; }
    }
    return out;
}

static std::string str(const std::string& v)  { return "\"" + esc(v) + "\""; }
static std::string b(bool v)                  { return v ? "true" : "false"; }
static std::string f(double v, int prec = 6) {
    if (!std::isfinite(v)) return "0.0";
    char buf[32]; snprintf(buf, sizeof(buf), "%.*f", prec, v); return buf;
}
static std::string i(int v)   { return std::to_string(v); }

} // namespace jw

// ─────────────────────────────────────────────
//  Minimal JSON parser (just what we need)
// ─────────────────────────────────────────────
namespace jp {

static void skip_ws(const char*& p) {
    while (*p && (*p==' '||*p=='\t'||*p=='\n'||*p=='\r')) ++p;
}

static std::string parse_string(const char*& p) {
    if (*p != '"') return "";
    ++p; // skip opening "
    std::string out;
    while (*p && *p != '"') {
        if (*p == '\\') { ++p; out += *p; }
        else             out += *p;
        ++p;
    }
    if (*p == '"') ++p;
    return out;
}

static double parse_number(const char*& p) {
    char* end;
    double v = strtod(p, &end);
    p = end;
    return v;
}

static bool parse_bool(const char*& p) {
    if (strncmp(p, "true", 4) == 0)  { p += 4; return true;  }
    if (strncmp(p, "false", 5) == 0) { p += 5; return false; }
    return false;
}

// Returns the value string for a key (shallow — no nesting) or ""
static std::string get_str(const char* json, const char* key) {
    std::string needle = std::string("\"") + key + "\"";
    const char* pos = strstr(json, needle.c_str());
    if (!pos) return "";
    pos += needle.size();
    skip_ws(pos);
    if (*pos != ':') return "";
    ++pos; skip_ws(pos);
    if (*pos != '"') return "";
    return parse_string(pos);
}

static double get_num(const char* json, const char* key, double def = 0.0) {
    std::string needle = std::string("\"") + key + "\"";
    const char* pos = strstr(json, needle.c_str());
    if (!pos) return def;
    pos += needle.size();
    skip_ws(pos);
    if (*pos != ':') return def;
    ++pos; skip_ws(pos);
    return parse_number(pos);
}

static bool get_bool(const char* json, const char* key, bool def = false) {
    std::string needle = std::string("\"") + key + "\"";
    const char* pos = strstr(json, needle.c_str());
    if (!pos) return def;
    pos += needle.size();
    skip_ws(pos);
    if (*pos != ':') return def;
    ++pos; skip_ws(pos);
    return parse_bool(pos);
}

} // namespace jp

// ─────────────────────────────────────────────
//  JsonBridge
// ─────────────────────────────────────────────
class JsonBridge {
public:
    JsonBridge(const std::string& state_path, const std::string& cmd_path)
        : m_state_path(state_path), m_cmd_path(cmd_path) {}

    // ── Write state ──────────────────────────
    void write_state(const BridgeState& s) {
        using namespace jw;

        std::ostringstream o;
        o << "{\n";
        o << "  \"timestamp\":"        << f(s.wall_time)          << ",\n";
        o << "  \"frame_number\":"     << i(s.render.frame_number) << ",\n";
        o << "  \"frame_time_ms\":"    << f(s.render.frame_time_ms,3) << ",\n";
        o << "  \"has_source\":"       << b(s.has_source)          << ",\n";
        o << "  \"source_path\":"      << str(s.source_path)       << ",\n";
        o << "  \"source_pos_sec\":"   << f(s.source_pos_sec)      << ",\n";
        o << "  \"source_dur_sec\":"   << f(s.source_dur_sec)      << ",\n";
        o << "  \"source_near_end\":"  << b(s.source_near_end)     << ",\n";
        o << "  \"active_passes\":"    << i(s.render.active_passes) << ",\n";
        // Crowd
        o << "  \"energy\":"           << f(s.crowd.energy)        << ",\n";
        o << "  \"density\":"          << f(s.crowd.density)       << ",\n";
        o << "  \"pulse\":"            << f(s.crowd.pulse)         << ",\n";
        o << "  \"frequency\":"        << f(s.crowd.frequency)     << ",\n";
        o << "  \"sentiment\":"        << f(s.crowd.sentiment)     << ",\n";
        // Passes array
        o << "  \"passes\": [\n";
        for (size_t pi = 0; pi < s.passes.size(); ++pi) {
            auto& p = s.passes[pi];
            o << "    {\"id\":" << str(p.id)
              << ",\"shader\":"  << str(p.shader)
              << ",\"enabled\":" << b(p.enabled) << "}";
            if (pi + 1 < s.passes.size()) o << ",";
            o << "\n";
        }
        o << "  ]\n}\n";

        atomic_write(m_state_path, o.str());
    }

    // ── Read commands ────────────────────────
    std::vector<EngineCommand> read_commands() {
        std::vector<EngineCommand> out;

        std::string raw = read_file(m_cmd_path);
        if (raw.empty()) return out;

        // Delete the file immediately after reading so we don't replay
        std::remove(m_cmd_path.c_str());

        // Parse the "commands" array
        const char* p = raw.c_str();
        const char* arr_start = strstr(p, "\"commands\"");
        if (!arr_start) return out;
        const char* bracket = strchr(arr_start, '[');
        if (!bracket) return out;
        ++bracket;

        // Walk through each {...} object in the array
        while (true) {
            jp::skip_ws(bracket);
            if (*bracket == ']' || *bracket == '\0') break;
            if (*bracket != '{') { ++bracket; continue; }

            // Find end of this object
            int depth = 1;
            const char* obj_start = bracket;
            ++bracket;
            while (*bracket && depth > 0) {
                if (*bracket == '{') ++depth;
                else if (*bracket == '}') --depth;
                ++bracket;
            }
            // bracket now points just past the closing }
            std::string obj(obj_start, bracket - obj_start);
            const char* o = obj.c_str();

            EngineCommand cmd = parse_command(o);
            out.push_back(cmd);

            jp::skip_ws(bracket);
            if (*bracket == ',') ++bracket;
        }

        return out;
    }

private:
    std::string m_state_path, m_cmd_path;

    // ── Parse one command object ────────────
    static EngineCommand parse_command(const char* o) {
        EngineCommand cmd;
        cmd.uniform_value = UniformValue::from_float(0.f);

        std::string type_str = jp::get_str(o, "type");

        if      (type_str == "ADD_PASS")     cmd.type = CommandType::ADD_PASS;
        else if (type_str == "REMOVE_PASS")  cmd.type = CommandType::REMOVE_PASS;
        else if (type_str == "ENABLE_PASS")  cmd.type = CommandType::ENABLE_PASS;
        else if (type_str == "SET_UNIFORM")  cmd.type = CommandType::SET_UNIFORM;
        else if (type_str == "SET_SOURCE")   cmd.type = CommandType::SET_SOURCE;
        else if (type_str == "REORDER_PASS") cmd.type = CommandType::REORDER_PASS;
        else                                 cmd.type = CommandType::ENABLE_PASS; // fallback

        cmd.pass_id      = jp::get_str(o, "pass_id");
        cmd.shader_name  = jp::get_str(o, "shader_name");
        cmd.source_path  = jp::get_str(o, "source_path");
        cmd.uniform_name = jp::get_str(o, "uniform_name");
        cmd.position     = (int)jp::get_num(o, "position", -1);
        cmd.enabled      = jp::get_bool(o, "enabled", true);

        // Uniform value
        std::string utype = jp::get_str(o, "uniform_type");
        // scalar value
        double uval = jp::get_num(o, "uniform_value", 0.0);

        if (utype == "int") {
            cmd.uniform_value = UniformValue::from_int((int)uval);
        } else if (utype == "vec2") {
            // Parse array [x, y] from raw text
            float v[4] = {0}; parse_float_array(o, "uniform_value", v, 2);
            cmd.uniform_value = UniformValue::from_vec2(v[0], v[1]);
        } else if (utype == "vec3") {
            float v[4] = {0}; parse_float_array(o, "uniform_value", v, 3);
            cmd.uniform_value = UniformValue::from_vec3(v[0], v[1], v[2]);
        } else if (utype == "vec4") {
            float v[4] = {0}; parse_float_array(o, "uniform_value", v, 4);
            cmd.uniform_value = UniformValue::from_vec4(v[0], v[1], v[2], v[3]);
        } else {
            // float (default)
            cmd.uniform_value = UniformValue::from_float((float)uval);
        }

        return cmd;
    }

    static void parse_float_array(const char* json, const char* key, float* out, int n) {
        std::string needle = std::string("\"") + key + "\"";
        const char* pos = strstr(json, needle.c_str());
        if (!pos) return;
        pos = strchr(pos, '[');
        if (!pos) return;
        ++pos;
        for (int i = 0; i < n; ++i) {
            jp::skip_ws(pos);
            char* end;
            out[i] = (float)strtod(pos, &end);
            pos = end;
            jp::skip_ws(pos);
            if (*pos == ',') ++pos;
        }
    }

    // ── File I/O ─────────────────────────────
    static void atomic_write(const std::string& path, const std::string& content) {
        std::string tmp = path + ".tmp";
        FILE* f = fopen(tmp.c_str(), "w");
        if (!f) return;
        fwrite(content.data(), 1, content.size(), f);
        fclose(f);
        std::rename(tmp.c_str(), path.c_str());
    }

    static std::string read_file(const std::string& path) {
        FILE* f = fopen(path.c_str(), "r");
        if (!f) return "";
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        if (sz <= 0) { fclose(f); return ""; }
        std::string s(sz, '\0');
        fread(&s[0], 1, sz, f);
        fclose(f);
        return s;
    }
};

// ─────────────────────────────────────────────
//  Helper: build BridgeState from live engine objects
// ─────────────────────────────────────────────
inline BridgeState make_bridge_state(
    const RenderStats&  stats,
    const CrowdState&   crowd,
    const VideoSource&  source,
    const RenderPipeline& pipeline,
    double wall_time)
{
    BridgeState s;
    s.render     = stats;
    s.crowd      = crowd;
    s.wall_time  = wall_time;
    s.has_source = source.is_open();
    s.source_path = source.path();

    // Query playback position and duration from FFmpeg
    // These are stored on VideoSource but not currently exposed — we expose them now
    // via a pair of accessors added to VideoSource (pos_sec / dur_sec).
    s.source_pos_sec  = source.pos_sec();
    s.source_dur_sec  = source.dur_sec();
    s.source_near_end = (s.source_dur_sec > 0 &&
                         s.source_pos_sec / s.source_dur_sec > 0.85);

    // Pass list — we need RenderPipeline to expose it
    for (auto& [id, shader, enabled] : pipeline.pass_list())
        s.passes.push_back({id, shader, enabled});

    return s;
}
