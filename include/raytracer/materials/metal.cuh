#ifndef RAYTRACER_MATERIALS_METAL_CUH
#define RAYTRACER_MATERIALS_METAL_CUH

#include "raytracer/materials/material.cuh"
#include "raytracer/core/random.cuh"

namespace rt {

__host__ __device__ inline Material create_metal(const Color& albedo, float fuzz) {
    Material mat(MaterialType::METAL, albedo);
    mat.fuzz = fuzz < 1.0f ? fuzz : 1.0f;
    return mat;
}

__device__ inline bool scatter_metal(
    const Material& mat,
    const Ray& r_in,
    const HitRecord& rec,
    Color& attenuation,
    Ray& scattered,
    curandState* rand_state
) {
    Vec3 reflected = reflect(r_in.direction(), rec.normal);
    reflected = unit_vector(reflected) + (mat.fuzz * random_unit_vector(rand_state));
    scattered = Ray(rec.p, reflected, r_in.time());
    attenuation = mat.albedo;
    return (dot(scattered.direction(), rec.normal) > 0);
}

// CPU version
inline bool scatter_metal_cpu(
    const Material& mat,
    const Ray& r_in,
    const HitRecord& rec,
    Color& attenuation,
    Ray& scattered,
    CPURandom& rng
) {
    Vec3 reflected = reflect(r_in.direction(), rec.normal);
    reflected = unit_vector(reflected) + (mat.fuzz * random_unit_vector(rng));
    scattered = Ray(rec.p, reflected, r_in.time());
    attenuation = mat.albedo;
    return (dot(scattered.direction(), rec.normal) > 0);
}

}

#endif
