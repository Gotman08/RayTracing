#ifndef RAYTRACER_TEXTURES_TEXTURE_CUH
#define RAYTRACER_TEXTURES_TEXTURE_CUH

#include "raytracer/core/vec3.cuh"

namespace rt {

class Texture {
public:
    Color color;

    __host__ __device__ Texture() : color(1, 1, 1) {}
    __host__ __device__ Texture(const Color& c) : color(c) {}

    __host__ __device__ Color value(float u, float v, const Point3& p) const {
        return color;
    }
};

}

#endif
