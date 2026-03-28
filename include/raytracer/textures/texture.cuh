#ifndef RAYTRACER_TEXTURES_TEXTURE_CUH
#define RAYTRACER_TEXTURES_TEXTURE_CUH

/** @file texture.cuh
 *  @brief Texture de base (couleur solide) */

#include "raytracer/core/vec3.cuh"

namespace rt {

/** @brief Texture simple : couleur constante */
class Texture {
public:
    Color color;

    __host__ __device__ Texture() : color(1, 1, 1) {}

    /** @brief Ctor avec couleur */
    __host__ __device__ Texture(const Color& c) : color(c) {}

    /** @brief Valeur de la texture (constante ici) */
    __host__ __device__ Color value(float u, float v, const Point3& p) const {
        return color;
    }
};

}

#endif
