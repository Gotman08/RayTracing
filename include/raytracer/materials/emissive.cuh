#ifndef RAYTRACER_MATERIALS_EMISSIVE_CUH
#define RAYTRACER_MATERIALS_EMISSIVE_CUH

#include "raytracer/materials/material.cuh"

namespace rt {

__host__ __device__ inline Material create_emissive(const Color& emit, float strength = 1.0f) {
    Material mat(MaterialType::EMISSIVE, Color(0, 0, 0));
    mat.emission = emit;
    mat.emission_strength = strength;
    return mat;
}

__device__ inline bool scatter_emissive(
    const Material& mat,
    const Ray& r_in,
    const HitRecord& rec,
    Color& attenuation,
    Ray& scattered,
    curandState* rand_state
) {
    // Emissive materials don't scatter
    return false;
}

} // namespace rt

#endif // RAYTRACER_MATERIALS_EMISSIVE_CUH
