#ifndef RAYTRACER_RENDERING_TONE_MAPPING_CUH
#define RAYTRACER_RENDERING_TONE_MAPPING_CUH

#include "raytracer/core/vec3.cuh"

namespace rt {

enum class ToneMapMode {
    NONE,
    REINHARD,
    REINHARD_EXTENDED,
    ACES,
    FILMIC
};

// Simple Reinhard tone mapping
__host__ __device__ inline Color tone_map_reinhard(const Color& hdr) {
    return Color(
        hdr.x / (1.0f + hdr.x),
        hdr.y / (1.0f + hdr.y),
        hdr.z / (1.0f + hdr.z)
    );
}

// Extended Reinhard with white point
__host__ __device__ inline Color tone_map_reinhard_extended(const Color& hdr, float white_point = 4.0f) {
    float wp2 = white_point * white_point;
    return Color(
        hdr.x * (1.0f + hdr.x / wp2) / (1.0f + hdr.x),
        hdr.y * (1.0f + hdr.y / wp2) / (1.0f + hdr.y),
        hdr.z * (1.0f + hdr.z / wp2) / (1.0f + hdr.z)
    );
}

// ACES Filmic Tone Mapping
__host__ __device__ inline Color tone_map_aces(const Color& hdr) {
    // ACES input transform (approximate)
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;

    auto aces_curve = [=](float x) {
        return (x * (a * x + b)) / (x * (c * x + d) + e);
    };

    return Color(
        fminf(1.0f, fmaxf(0.0f, aces_curve(hdr.x))),
        fminf(1.0f, fmaxf(0.0f, aces_curve(hdr.y))),
        fminf(1.0f, fmaxf(0.0f, aces_curve(hdr.z)))
    );
}

// Filmic (Uncharted 2) Tone Mapping
__host__ __device__ inline Color tone_map_filmic(const Color& hdr) {
    auto filmic_curve = [](float x) {
        float A = 0.15f;
        float B = 0.50f;
        float C = 0.10f;
        float D = 0.20f;
        float E = 0.02f;
        float F = 0.30f;
        return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
    };

    float white_scale = 1.0f / filmic_curve(11.2f);

    return Color(
        filmic_curve(hdr.x) * white_scale,
        filmic_curve(hdr.y) * white_scale,
        filmic_curve(hdr.z) * white_scale
    );
}

// Apply tone mapping based on mode
__host__ __device__ inline Color apply_tone_mapping(const Color& hdr, ToneMapMode mode) {
    switch (mode) {
        case ToneMapMode::REINHARD:
            return tone_map_reinhard(hdr);
        case ToneMapMode::REINHARD_EXTENDED:
            return tone_map_reinhard_extended(hdr);
        case ToneMapMode::ACES:
            return tone_map_aces(hdr);
        case ToneMapMode::FILMIC:
            return tone_map_filmic(hdr);
        case ToneMapMode::NONE:
        default:
            return hdr;
    }
}

// Gamma correction (linear to sRGB)
__host__ __device__ inline Color gamma_correct(const Color& linear, float gamma = 2.2f) {
    float inv_gamma = 1.0f / gamma;
    return Color(
        powf(fmaxf(0.0f, linear.x), inv_gamma),
        powf(fmaxf(0.0f, linear.y), inv_gamma),
        powf(fmaxf(0.0f, linear.z), inv_gamma)
    );
}

// Exposure adjustment
__host__ __device__ inline Color apply_exposure(const Color& hdr, float exposure) {
    float factor = powf(2.0f, exposure);
    return hdr * factor;
}

} // namespace rt

#endif // RAYTRACER_RENDERING_TONE_MAPPING_CUH
