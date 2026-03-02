#ifndef RAYTRACER_ENVIRONMENT_SKY_CUH
#define RAYTRACER_ENVIRONMENT_SKY_CUH

#include "raytracer/core/vec3.cuh"

namespace rt {

// Procedural sky background
class Sky {
public:
    Color horizon_color;
    Color zenith_color;
    float sun_intensity;
    Vec3 sun_direction;
    float sun_size;

    __host__ __device__ Sky()
        : horizon_color(1.0f, 1.0f, 1.0f),
          zenith_color(0.5f, 0.7f, 1.0f),
          sun_intensity(0.0f),
          sun_direction(0, 1, 0),
          sun_size(0.01f) {}

    __host__ __device__ Sky(const Color& horizon, const Color& zenith)
        : horizon_color(horizon), zenith_color(zenith),
          sun_intensity(0.0f), sun_direction(0, 1, 0), sun_size(0.01f) {}

    __host__ __device__ void set_sun(const Vec3& dir, float intensity, float size = 0.01f) {
        sun_direction = unit_vector(dir);
        sun_intensity = intensity;
        sun_size = size;
    }

    __host__ __device__ Color get_color(const Vec3& direction) const {
        Vec3 unit_dir = unit_vector(direction);

        // Sky gradient based on y component
        float t = 0.5f * (unit_dir.y + 1.0f);
        Color sky = (1.0f - t) * horizon_color + t * zenith_color;

        // Add sun if enabled
        if (sun_intensity > 0.0f) {
            float sun_dot = dot(unit_dir, sun_direction);
            if (sun_dot > (1.0f - sun_size)) {
                float sun_factor = (sun_dot - (1.0f - sun_size)) / sun_size;
                sun_factor = sun_factor * sun_factor;
                sky = sky + sun_intensity * sun_factor * Color(1.0f, 0.95f, 0.8f);
            }
        }

        return sky;
    }
};

// Simple gradient sky
__host__ __device__ inline Color sky_gradient(const Vec3& direction) {
    Vec3 unit_dir = unit_vector(direction);
    float t = 0.5f * (unit_dir.y + 1.0f);
    return (1.0f - t) * Color(1.0f, 1.0f, 1.0f) + t * Color(0.5f, 0.7f, 1.0f);
}

// Black background
__host__ __device__ inline Color sky_black(const Vec3& direction) {
    return Color(0.0f, 0.0f, 0.0f);
}

} // namespace rt

#endif // RAYTRACER_ENVIRONMENT_SKY_CUH
