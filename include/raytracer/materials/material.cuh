#ifndef RAYTRACER_MATERIALS_MATERIAL_CUH
#define RAYTRACER_MATERIALS_MATERIAL_CUH

#include "raytracer/core/vec3.cuh"
#include "raytracer/core/ray.cuh"
#include "raytracer/textures/solid_color.cuh"
#include "raytracer/textures/checker.cuh"
#include "raytracer/textures/noise.cuh"

namespace rt {

enum class MaterialType {
    LAMBERTIAN,
    METAL,
    DIELECTRIC,
    EMISSIVE,
    ISOTROPIC
};

class Material {
public:
    MaterialType type;
    Color albedo;
    float fuzz;
    float ior;
    Color emission;
    float emission_strength;

    // Texture support
    TextureType tex_type;
    SolidColor solid_tex;
    CheckerTexture checker_tex;
    NoiseTexture noise_tex;

    __host__ __device__ Material()
        : type(MaterialType::LAMBERTIAN), albedo(0.5f, 0.5f, 0.5f),
          fuzz(0), ior(1.0f), emission(0, 0, 0), emission_strength(0),
          tex_type(TextureType::SOLID_COLOR) {}

    __host__ __device__ Material(MaterialType t, const Color& a)
        : type(t), albedo(a), fuzz(0), ior(1.0f),
          emission(0, 0, 0), emission_strength(0),
          tex_type(TextureType::SOLID_COLOR), solid_tex(a) {}

    __host__ __device__ Color get_albedo(float u, float v, const Point3& p) const {
        switch (tex_type) {
            case TextureType::CHECKER:
                return checker_tex.value(u, v, p);
            case TextureType::NOISE:
                return noise_tex.value(u, v, p);
            case TextureType::SOLID_COLOR:
            default:
                return solid_tex.value(u, v, p);
        }
    }

    __host__ __device__ Color emitted(float u, float v, const Point3& p) const {
        if (type == MaterialType::EMISSIVE) {
            return emission * emission_strength;
        }
        return Color(0, 0, 0);
    }
};

} // namespace rt

#endif // RAYTRACER_MATERIALS_MATERIAL_CUH
