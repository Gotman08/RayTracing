#ifndef RAYTRACER_RENDERING_TONE_MAPPING_CUH
#define RAYTRACER_RENDERING_TONE_MAPPING_CUH

/** @file tone_mapping.cuh
 * @brief Tone mapping Reinhard + correction gamma (host/device) */

#include "raytracer/core/vec3.cuh"

namespace rt {

/** @brief Reinhard simple : x/(1+x) par canal RGB */
__host__ __device__ inline Color apply_tone_mapping(const Color& hdr) {
    return Color(
        hdr.x / (1.0f + hdr.x),
        hdr.y / (1.0f + hdr.y),
        hdr.z / (1.0f + hdr.z)
    );
}

/** @brief Correction gamma lineaire -> sRGB (defaut 2.2) */
__host__ __device__ inline Color gamma_correct(const Color& linear, float gamma = 2.2f) {
    float inv_gamma = 1.0f / gamma;
    return Color(
        powf(fmaxf(0.0f, linear.x), inv_gamma),
        powf(fmaxf(0.0f, linear.y), inv_gamma),
        powf(fmaxf(0.0f, linear.z), inv_gamma)
    );
}

/** @brief Luminance perceptuelle Rec.709 */
__host__ __device__ inline float luminance(const Color& c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

/** @brief Reinhard etendu adaptatif (key = gris moyen 18%) */
__host__ __device__ inline Color apply_adaptive_tone_mapping(
    const Color& hdr, float avg_lum, float key = 0.18f
) {
    // eviter la division par zero si la scene est noire
    float safe_avg = fmaxf(avg_lum, 1e-6f);
    float scale = key / safe_avg;

    Color scaled(hdr.x * scale, hdr.y * scale, hdr.z * scale);
    return Color(
        scaled.x / (1.0f + scaled.x),
        scaled.y / (1.0f + scaled.y),
        scaled.z / (1.0f + scaled.z)
    );
}

}

#endif
