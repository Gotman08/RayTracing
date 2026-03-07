#ifndef RAYTRACER_TEXTURES_SOLID_COLOR_CUH
#define RAYTRACER_TEXTURES_SOLID_COLOR_CUH

#include "raytracer/core/vec3.cuh"

namespace rt {

class SolidColor {
public:
    Color albedo;

    __host__ __device__ SolidColor() : albedo(0, 0, 0) {}

    __host__ __device__ SolidColor(const Color& c) : albedo(c) {}

    __host__ __device__ SolidColor(float r, float g, float b) : albedo(r, g, b) {}

    __host__ __device__ Color value(float u, float v, const Point3& p) const {
        return albedo;
    }
};

}

#endif
