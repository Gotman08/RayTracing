#ifndef RAYTRACER_MATERIALS_ISOTROPIC_CUH
#define RAYTRACER_MATERIALS_ISOTROPIC_CUH

#include "raytracer/materials/material.cuh"

namespace rt {

// Isotropic material for participating media (fog, smoke, etc.)
__host__ __device__ inline Material create_isotropic(const Color& albedo) {
    Material mat(MaterialType::ISOTROPIC, albedo);
    return mat;
}

__device__ inline bool scatter_isotropic(
    const Material& mat,
    const Ray& r_in,
    const HitRecord& rec,
    Color& attenuation,
    Ray& scattered,
    curandState* rand_state
) {
    // Scatter in random direction (isotropic)
    scattered = Ray(rec.p, random_unit_vector(rand_state), r_in.time());
    attenuation = mat.albedo;
    return true;
}

} // namespace rt

#endif // RAYTRACER_MATERIALS_ISOTROPIC_CUH
