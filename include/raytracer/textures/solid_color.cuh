#ifndef RAYTRACER_TEXTURES_SOLID_COLOR_CUH
#define RAYTRACER_TEXTURES_SOLID_COLOR_CUH

/** @file solid_color.cuh
 *  @brief Texture couleur unie (albedo constant) */

#include "raytracer/core/vec3.cuh"

namespace rt {

/** @brief Couleur unie, retourne toujours le meme albedo */
class SolidColor {
public:
    Color albedo;

    __host__ __device__ SolidColor() : albedo(0, 0, 0) {}

    /** @brief Ctor depuis Color */
    __host__ __device__ SolidColor(const Color& c) : albedo(c) {}

    /** @brief Ctor depuis composantes RGB */
    __host__ __device__ SolidColor(float r, float g, float b) : albedo(r, g, b) {}

    /** @brief Retourne l'albedo (u, v, p ignores) */
    __host__ __device__ Color value(float u, float v, const Point3& p) const {
        return albedo;
    }
};

}

#endif
