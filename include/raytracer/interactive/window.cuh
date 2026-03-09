#ifndef RAYTRACER_INTERACTIVE_WINDOW_CUH
#define RAYTRACER_INTERACTIVE_WINDOW_CUH

#ifdef ENABLE_INTERACTIVE

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cstdio>

namespace rt {

class Window {
public:
    GLFWwindow* handle;
    int width, height;

    Window() : handle(nullptr), width(0), height(0) {}

    bool initialize(int w, int h, const char* title) {
        width = w;
        height = h;

        if (!glfwInit()) {
            fprintf(stderr, "Failed to initialize GLFW\n");
            return false;
        }

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        handle = glfwCreateWindow(width, height, title, nullptr, nullptr);
        if (!handle) {
            fprintf(stderr, "Failed to create GLFW window\n");
            glfwTerminate();
            return false;
        }

        glfwMakeContextCurrent(handle);

        // Initialize GLEW
        glewExperimental = GL_TRUE;
        GLenum glew_err = glewInit();
        if (glew_err != GLEW_OK) {
            fprintf(stderr, "Failed to initialize GLEW: %s\n", glewGetErrorString(glew_err));
            glfwDestroyWindow(handle);
            glfwTerminate();
            return false;
        }

        // Disable V-Sync for maximum framerate
        glfwSwapInterval(0);

        // Capture mouse
        glfwSetInputMode(handle, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

        // Setup viewport
        glViewport(0, 0, width, height);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(-1, 1, -1, 1, -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        return true;
    }

    bool should_close() const {
        return glfwWindowShouldClose(handle);
    }

    void swap_buffers() {
        glfwSwapBuffers(handle);
    }

    void set_title(const char* title) {
        glfwSetWindowTitle(handle, title);
    }

    void cleanup() {
        if (handle) {
            glfwDestroyWindow(handle);
            handle = nullptr;
        }
        glfwTerminate();
    }

    ~Window() {
        // Note: cleanup() should be called explicitly before CUDA cleanup
    }
};

} // namespace rt

#endif // ENABLE_INTERACTIVE
#endif // RAYTRACER_INTERACTIVE_WINDOW_CUH
