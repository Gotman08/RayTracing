#ifndef RAYTRACER_MATERIALS_DIELECTRIC_CUH
#define RAYTRACER_MATERIALS_DIELECTRIC_CUH

/** @file dielectric.cuh
 *  @brief Dielectrique : Snell + Fresnel/Schlick */

#include "raytracer/materials/material.cuh"
#include "raytracer/core/random.cuh"

namespace rt {

/** @brief Fabrique un dielectrique (albedo blanc + ior) */
__host__ __device__ inline Material create_dielectric(float ior) {
    Material mat(MaterialType::DIELECTRIC, Color(1, 1, 1));
    mat.ior = ior;
    return mat;
}

/** @brief Approx. de Schlick pour la reflectance Fresnel */
__host__ __device__ inline float reflectance(float cosine, float ior) {
    float r0 = (1.0f - ior) / (1.0f + ior);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * powf((1.0f - cosine), 5.0f);
}

/** @brief Scatter dielectrique GPU : reflexion totale ou refraction */
__device__ inline bool scatter_dielectric(
    const Material& mat,
    const Ray& r_in,
    const HitRecord& rec,
    Color& attenuation,
    Ray& scattered,
    curandState* rand_state
) {
    attenuation = Color(1.0f, 1.0f, 1.0f);
    float ri = rec.front_face ? (1.0f / mat.ior) : mat.ior;

    Vec3 unit_direction = unit_vector(r_in.direction());
    float cos_theta = fminf(dot(-unit_direction, rec.normal), 1.0f);
    float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

    bool cannot_refract = ri * sin_theta > 1.0f;
    Vec3 direction;

    if (cannot_refract || reflectance(cos_theta, ri) > curand_uniform(rand_state)) {
        direction = reflect(unit_direction, rec.normal);
    } else {
        direction = refract(unit_direction, rec.normal, ri);
    }

    scattered = Ray(rec.p, direction, r_in.time());
    return true;
}

/** @brief Scatter dielectrique CPU (meme algo, rng CPU) */
inline bool scatter_dielectric_cpu(
    const Material& mat,
    const Ray& r_in,
    const HitRecord& rec,
    Color& attenuation,
    Ray& scattered,
    CPURandom& rng
) {
    attenuation = Color(1.0f, 1.0f, 1.0f);
    float ri = rec.front_face ? (1.0f / mat.ior) : mat.ior;

    Vec3 unit_direction = unit_vector(r_in.direction());
    float cos_theta = fminf(dot(-unit_direction, rec.normal), 1.0f);
    float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

    bool cannot_refract = ri * sin_theta > 1.0f;
    Vec3 direction;

    if (cannot_refract || reflectance(cos_theta, ri) > rng()) {
        direction = reflect(unit_direction, rec.normal);
    } else {
        direction = refract(unit_direction, rec.normal, ri);
    }

    scattered = Ray(rec.p, direction, r_in.time());
    return true;
}

}

#endif
