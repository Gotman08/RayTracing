#ifndef RAYTRACER_RENDERING_TONE_MAPPING_CUH
#define RAYTRACER_RENDERING_TONE_MAPPING_CUH

#include "raytracer/core/vec3.cuh"

namespace rt {

__host__ __device__ inline Color apply_tone_mapping(const Color& hdr) {
    return Color(
        hdr.x / (1.0f + hdr.x),
        hdr.y / (1.0f + hdr.y),
        hdr.z / (1.0f + hdr.z)
    );
}

__host__ __device__ inline Color gamma_correct(const Color& linear, float gamma = 2.2f) {
    float inv_gamma = 1.0f / gamma;
    return Color(
        powf(fmaxf(0.0f, linear.x), inv_gamma),
        powf(fmaxf(0.0f, linear.y), inv_gamma),
        powf(fmaxf(0.0f, linear.z), inv_gamma)
    );
}

}

#endif
