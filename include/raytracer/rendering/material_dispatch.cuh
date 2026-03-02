/**
 * Material Scatter Dispatch
 * Central dispatch function for all material types
 */

#ifndef RAYTRACER_RENDERING_MATERIAL_DISPATCH_CUH
#define RAYTRACER_RENDERING_MATERIAL_DISPATCH_CUH

#include "raytracer/core/vec3.cuh"
#include "raytracer/core/ray.cuh"
#include "raytracer/core/hit_record.cuh"
#include "raytracer/materials/material.cuh"
#include "raytracer/materials/lambertian.cuh"
#include "raytracer/materials/metal.cuh"
#include "raytracer/materials/dielectric.cuh"
#include "raytracer/materials/emissive.cuh"
#include "raytracer/materials/isotropic.cuh"

namespace rt {

/**
 * Dispatch scatter function based on material type
 * @return true if ray scattered, false if absorbed
 */
__device__ inline bool scatter(
    const Material& mat,
    const Ray& r_in,
    const HitRecord& rec,
    Color& attenuation,
    Ray& scattered,
    curandState* rand_state
) {
    switch (mat.type) {
        case MaterialType::LAMBERTIAN:
            return scatter_lambertian(mat, r_in, rec, attenuation, scattered, rand_state);
        case MaterialType::METAL:
            return scatter_metal(mat, r_in, rec, attenuation, scattered, rand_state);
        case MaterialType::DIELECTRIC:
            return scatter_dielectric(mat, r_in, rec, attenuation, scattered, rand_state);
        case MaterialType::EMISSIVE:
            return scatter_emissive(mat, r_in, rec, attenuation, scattered, rand_state);
        case MaterialType::ISOTROPIC:
            return scatter_isotropic(mat, r_in, rec, attenuation, scattered, rand_state);
        default:
            return false;
    }
}

} // namespace rt

#endif // RAYTRACER_RENDERING_MATERIAL_DISPATCH_CUH
