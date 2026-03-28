#ifndef RAYTRACER_RENDERING_MATERIAL_DISPATCH_CUH
#define RAYTRACER_RENDERING_MATERIAL_DISPATCH_CUH

/** @file material_dispatch.cuh
 * @brief Dispatch scatter() selon le type de materiau (GPU + CPU) */

#include "raytracer/core/vec3.cuh"
#include "raytracer/core/ray.cuh"
#include "raytracer/core/hit_record.cuh"
#include "raytracer/materials/material.cuh"
#include "raytracer/materials/lambertian.cuh"
#include "raytracer/materials/metal.cuh"
#include "raytracer/materials/dielectric.cuh"

namespace rt {

/** @brief Scatter GPU - switch sur MaterialType (lambertian/metal/dielectric) */
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
        default:
            return false;
    }
}

/** @brief Scatter CPU - meme dispatch, utilise CPURandom au lieu de curand */
inline bool scatter_cpu(
    const Material& mat,
    const Ray& r_in,
    const HitRecord& rec,
    Color& attenuation,
    Ray& scattered,
    CPURandom& rng
) {
    switch (mat.type) {
        case MaterialType::LAMBERTIAN:
            return scatter_lambertian_cpu(mat, r_in, rec, attenuation, scattered, rng);
        case MaterialType::METAL:
            return scatter_metal_cpu(mat, r_in, rec, attenuation, scattered, rng);
        case MaterialType::DIELECTRIC:
            return scatter_dielectric_cpu(mat, r_in, rec, attenuation, scattered, rng);
        default:
            return false;
    }
}

}

#endif
