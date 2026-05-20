#pragma once
/*
 * frame_export.h
 * Dumps current GL framebuffer to a raw PPM file (readable by PIL/Pillow).
 * Atomic write via .tmp rename.
 */
#include <string>
#include <vector>
#include <cstdio>
#include <GL/glew.h>

class FrameExporter {
public:
    FrameExporter(const std::string& path = "vj_frame.ppm", int every_n = 30)
        : m_path(path), m_every(every_n) {}

    void maybe_export(int frame_number, int width, int height) {
        if (m_every > 0 && frame_number % m_every != 0) return;
        export_now(width, height);
    }

    void export_now(int width, int height) {
        if (width <= 0 || height <= 0) return;
        m_buf.resize((size_t)width * height * 3);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, m_buf.data());

        // Flip Y (GL origin is bottom-left)
        const int row = width * 3;
        for (int y = 0; y < height / 2; ++y) {
            uint8_t* a = m_buf.data() + y * row;
            uint8_t* b = m_buf.data() + (height - 1 - y) * row;
            for (int x = 0; x < row; ++x) std::swap(a[x], b[x]);
        }

        std::string tmp = m_path + ".tmp";
        FILE* f = fopen(tmp.c_str(), "wb");
        if (!f) return;
        fprintf(f, "P6\n%d %d\n255\n", width, height);
        fwrite(m_buf.data(), 1, m_buf.size(), f);
        fclose(f);
        std::rename(tmp.c_str(), m_path.c_str());
    }

private:
    std::string          m_path;
    int                  m_every;
    std::vector<uint8_t> m_buf;
};
