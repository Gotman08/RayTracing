#ifndef RAYTRACER_INTERACTIVE_INPUT_HANDLER_CUH
#define RAYTRACER_INTERACTIVE_INPUT_HANDLER_CUH

#ifdef ENABLE_INTERACTIVE

#include <GLFW/glfw3.h>

namespace rt {

struct InputState {
    // Movement keys (WASD + QE for up/down)
    bool forward = false;
    bool backward = false;
    bool left = false;
    bool right = false;
    bool up = false;
    bool down = false;

    // Speed modifier (shift)
    bool fast = false;

    // Mouse state
    double mouse_x = 0.0;
    double mouse_y = 0.0;
    double last_mouse_x = 0.0;
    double last_mouse_y = 0.0;
    double mouse_delta_x = 0.0;
    double mouse_delta_y = 0.0;
    bool first_mouse = true;
    bool mouse_captured = true;

    // Control state
    bool should_close = false;
    bool screenshot_requested = false;
    bool reset_accumulation = false;

    void update_mouse_delta() {
        if (first_mouse) {
            last_mouse_x = mouse_x;
            last_mouse_y = mouse_y;
            first_mouse = false;
        }
        mouse_delta_x = mouse_x - last_mouse_x;
        mouse_delta_y = last_mouse_y - mouse_y; // Inverted Y
        last_mouse_x = mouse_x;
        last_mouse_y = mouse_y;
    }

    bool has_movement() const {
        return forward || backward || left || right || up || down ||
               (mouse_captured && (mouse_delta_x != 0.0 || mouse_delta_y != 0.0));
    }

    void clear_deltas() {
        mouse_delta_x = 0.0;
        mouse_delta_y = 0.0;
        reset_accumulation = false;
        screenshot_requested = false;
    }
};

class InputHandler {
public:
    InputState state;

    void setup_callbacks(GLFWwindow* window) {
        glfwSetWindowUserPointer(window, this);

        glfwSetKeyCallback(window, [](GLFWwindow* win, int key, int scancode, int action, int mods) {
            auto* handler = static_cast<InputHandler*>(glfwGetWindowUserPointer(win));
            handler->key_callback(win, key, action);
        });

        glfwSetCursorPosCallback(window, [](GLFWwindow* win, double x, double y) {
            auto* handler = static_cast<InputHandler*>(glfwGetWindowUserPointer(win));
            handler->state.mouse_x = x;
            handler->state.mouse_y = y;
        });

        glfwSetMouseButtonCallback(window, [](GLFWwindow* win, int button, int action, int mods) {
            auto* handler = static_cast<InputHandler*>(glfwGetWindowUserPointer(win));
            if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
                handler->toggle_mouse_capture(win);
            }
        });
    }

    void poll_events() {
        glfwPollEvents();
        state.update_mouse_delta();
    }

private:
    void key_callback(GLFWwindow* window, int key, int action) {
        bool pressed = (action == GLFW_PRESS || action == GLFW_REPEAT);

        switch (key) {
            case GLFW_KEY_W:           state.forward = pressed; break;
            case GLFW_KEY_S:           state.backward = pressed; break;
            case GLFW_KEY_A:           state.left = pressed; break;
            case GLFW_KEY_D:           state.right = pressed; break;
            case GLFW_KEY_Q:
            case GLFW_KEY_SPACE:       state.up = pressed; break;
            case GLFW_KEY_E:
            case GLFW_KEY_LEFT_CONTROL: state.down = pressed; break;
            case GLFW_KEY_LEFT_SHIFT:  state.fast = pressed; break;
            case GLFW_KEY_ESCAPE:
                if (action == GLFW_PRESS) state.should_close = true;
                break;
            case GLFW_KEY_P:
            case GLFW_KEY_F12:
                if (action == GLFW_PRESS) state.screenshot_requested = true;
                break;
            case GLFW_KEY_R:
                if (action == GLFW_PRESS) state.reset_accumulation = true;
                break;
        }
    }

    void toggle_mouse_capture(GLFWwindow* window) {
        state.mouse_captured = !state.mouse_captured;
        glfwSetInputMode(window, GLFW_CURSOR,
            state.mouse_captured ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
        state.first_mouse = true;
    }
};

} // namespace rt

#endif // ENABLE_INTERACTIVE
#endif // RAYTRACER_INTERACTIVE_INPUT_HANDLER_CUH
