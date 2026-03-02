#ifndef RAYTRACER_TEXTURES_SOLID_COLOR_CUH
#define RAYTRACER_TEXTURES_SOLID_COLOR_CUH

#include "raytracer/textures/texture.cuh"

namespace rt {

class SolidColor : public Texture {
public:
    Color albedo;

    __host__ __device__ SolidColor() : Texture(TextureType::SOLID_COLOR), albedo(0, 0, 0) {}

    __host__ __device__ SolidColor(const Color& c)
        : Texture(TextureType::SOLID_COLOR), albedo(c) {}

    __host__ __device__ SolidColor(float r, float g, float b)
        : Texture(TextureType::SOLID_COLOR), albedo(r, g, b) {}

    __host__ __device__ Color value(float u, float v, const Point3& p) const {
        return albedo;
    }
};

} // namespace rt

#endif // RAYTRACER_TEXTURES_SOLID_COLOR_CUH
