#ifndef RAYTRACER_MATERIALS_MATERIAL_CUH
#define RAYTRACER_MATERIALS_MATERIAL_CUH

#include "raytracer/core/vec3.cuh"
#include "raytracer/core/ray.cuh"

namespace rt {

enum class MaterialType {
    LAMBERTIAN,
    METAL,
    DIELECTRIC
};

class Material {
public:
    MaterialType type;
    Color albedo;
    float fuzz;
    float ior;

    __host__ __device__ Material()
        : type(MaterialType::LAMBERTIAN), albedo(0.5f, 0.5f, 0.5f),
          fuzz(0), ior(1.5f) {}

    __host__ __device__ Material(MaterialType t, const Color& a)
        : type(t), albedo(a), fuzz(0), ior(1.5f) {}

    __host__ __device__ Material(MaterialType t, const Color& a, float f)
        : type(t), albedo(a), fuzz(f), ior(1.5f) {}

    __host__ __device__ static Material make_dielectric(float index_of_refraction) {
        Material m;
        m.type = MaterialType::DIELECTRIC;
        m.albedo = Color(1, 1, 1);
        m.ior = index_of_refraction;
        return m;
    }

    __host__ __device__ Color get_albedo(float u, float v, const Point3& p) const {
        return albedo;
    }

    __host__ __device__ Color emitted(float u, float v, const Point3& p) const {
        return Color(0, 0, 0);
    }
};

}

#endif
