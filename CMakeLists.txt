cmake_minimum_required(VERSION 3.16)
project(vjengine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── Find packages ───────────────────────────────────────
find_package(PkgConfig REQUIRED)

pkg_check_modules(SDL2     REQUIRED sdl2)
pkg_check_modules(GLEW     REQUIRED glew)
pkg_check_modules(AVCODEC  REQUIRED libavcodec)
pkg_check_modules(AVFORMAT REQUIRED libavformat)
pkg_check_modules(AVUTIL   REQUIRED libavutil)
pkg_check_modules(SWSCALE  REQUIRED libswscale)

find_package(OpenGL REQUIRED)

# ─── Sources ─────────────────────────────────────────────
set(SOURCES
    src/main.cpp
    src/video_source.cpp
    src/shader_manager.cpp
    src/render_pipeline.cpp
)

add_executable(vjengine ${SOURCES})

# ─── Include dirs ────────────────────────────────────────
target_include_directories(vjengine PRIVATE
    src/
    ${SDL2_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
    ${AVCODEC_INCLUDE_DIRS}
    ${AVFORMAT_INCLUDE_DIRS}
    ${AVUTIL_INCLUDE_DIRS}
    ${SWSCALE_INCLUDE_DIRS}
)

# ─── Link libs ───────────────────────────────────────────
target_link_libraries(vjengine PRIVATE
    ${SDL2_LIBRARIES}
    ${GLEW_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${AVCODEC_LIBRARIES}
    ${AVFORMAT_LIBRARIES}
    ${AVUTIL_LIBRARIES}
    ${SWSCALE_LIBRARIES}
    pthread
)

target_compile_options(vjengine PRIVATE
    ${SDL2_CFLAGS_OTHER}
    -O2 -Wall -Wextra
)

# ─── Copy shaders next to binary ─────────────────────────
add_custom_command(TARGET vjengine POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/shaders
        $<TARGET_FILE_DIR:vjengine>/shaders
    COMMENT "Copying shaders"
)
