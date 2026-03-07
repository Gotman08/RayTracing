#ifndef RAYTRACER_MATERIALS_LAMBERTIAN_CUH
#define RAYTRACER_MATERIALS_LAMBERTIAN_CUH

#include "raytracer/materials/material.cuh"
#include "raytracer/core/random.cuh"

namespace rt {

__host__ __device__ inline Material create_lambertian(const Color& albedo) {
    Material mat(MaterialType::LAMBERTIAN, albedo);
    return mat;
}

__device__ inline bool scatter_lambertian(
    const Material& mat,
    const Ray& r_in,
    const HitRecord& rec,
    Color& attenuation,
    Ray& scattered,
    curandState* rand_state
) {
    Vec3 scatter_direction = rec.normal + random_unit_vector(rand_state);

    if (scatter_direction.near_zero())
        scatter_direction = rec.normal;

    scattered = Ray(rec.p, scatter_direction, r_in.time());
    attenuation = mat.albedo;
    return true;
}

// CPU version
inline bool scatter_lambertian_cpu(
    const Material& mat,
    const Ray& r_in,
    const HitRecord& rec,
    Color& attenuation,
    Ray& scattered,
    CPURandom& rng
) {
    Vec3 scatter_direction = rec.normal + random_unit_vector(rng);

    if (scatter_direction.near_zero())
        scatter_direction = rec.normal;

    scattered = Ray(rec.p, scatter_direction, r_in.time());
    attenuation = mat.albedo;
    return true;
}

}

#endif
