#ifndef RAYTRACER_INTERACTIVE_CAMERA_CONTROLLER_CUH
#define RAYTRACER_INTERACTIVE_CAMERA_CONTROLLER_CUH

#ifdef ENABLE_INTERACTIVE

#include "raytracer/core/vec3.cuh"
#include "raytracer/camera/camera.cuh"
#include "raytracer/interactive/input_handler.cuh"
#include <cmath>

namespace rt {

class CameraController {
public:
    // Camera position and orientation
    Point3 position;
    float yaw;          // Rotation around Y axis (radians)
    float pitch;        // Rotation around X axis (radians)

    // Camera parameters
    float vfov;
    float aperture;
    float focus_dist;

    // Control parameters
    float move_speed;
    float mouse_sensitivity;

    // State tracking
    bool camera_changed;

    CameraController()
        : position(0, 0, 0), yaw(0), pitch(0),
          vfov(20.0f), aperture(0.0f), focus_dist(10.0f),
          move_speed(5.0f), mouse_sensitivity(0.002f),
          camera_changed(true) {}

    // Initialize from existing camera parameters
    void initialize(const Point3& lookfrom, const Point3& lookat,
                   float fov, float ap = 0.0f, float fd = 10.0f) {
        position = lookfrom;
        vfov = fov;
        aperture = ap;
        focus_dist = fd;

        // Compute yaw and pitch from look direction
        Vec3 look_dir = unit_vector(lookat - lookfrom);
        yaw = atan2f(look_dir.x, -look_dir.z);
        pitch = asinf(fmaxf(-0.99f, fminf(0.99f, look_dir.y)));

        camera_changed = true;
    }

    // Process input and update camera state
    bool update(const InputState& input, float delta_time) {
        camera_changed = false;

        // Handle mouse look (only when captured)
        if (input.mouse_captured) {
            if (input.mouse_delta_x != 0.0 || input.mouse_delta_y != 0.0) {
                yaw += static_cast<float>(input.mouse_delta_x) * mouse_sensitivity;
                pitch += static_cast<float>(input.mouse_delta_y) * mouse_sensitivity;

                // Clamp pitch to prevent flipping
                constexpr float MAX_PITCH = 1.5f; // ~86 degrees
                pitch = fmaxf(-MAX_PITCH, fminf(MAX_PITCH, pitch));

                camera_changed = true;
            }
        }

        // Compute forward and right vectors
        Vec3 forward = get_forward();
        Vec3 right = get_right();
        Vec3 up(0, 1, 0);

        // Handle movement
        Vec3 velocity(0, 0, 0);
        if (input.forward)  velocity = velocity + forward;
        if (input.backward) velocity = velocity - forward;
        if (input.right)    velocity = velocity + right;
        if (input.left)     velocity = velocity - right;
        if (input.up)       velocity = velocity + up;
        if (input.down)     velocity = velocity - up;

        if (velocity.length_squared() > 0.0f) {
            float speed = input.fast ? move_speed * 3.0f : move_speed;
            position = position + unit_vector(velocity) * speed * delta_time;
            camera_changed = true;
        }

        return camera_changed;
    }

    // Get forward direction from yaw/pitch
    Vec3 get_forward() const {
        return Vec3(
            sinf(yaw) * cosf(pitch),
            sinf(pitch),
            -cosf(yaw) * cosf(pitch)
        );
    }

    // Get right direction
    Vec3 get_right() const {
        return Vec3(cosf(yaw), 0, sinf(yaw));
    }

    // Get look-at point
    Point3 get_lookat() const {
        return position + get_forward() * focus_dist;
    }

    // Build Camera struct for kernel
    Camera build_camera(int width, int height) const {
        Camera camera;
        camera.initialize(
            width, height,
            position,
            get_lookat(),
            Vec3(0, 1, 0),
            vfov, aperture, focus_dist
        );
        return camera;
    }
};

} // namespace rt

#endif // ENABLE_INTERACTIVE
#endif // RAYTRACER_INTERACTIVE_CAMERA_CONTROLLER_CUH
