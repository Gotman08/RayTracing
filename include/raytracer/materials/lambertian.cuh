#ifndef RAYTRACER_MATERIALS_LAMBERTIAN_CUH
#define RAYTRACER_MATERIALS_LAMBERTIAN_CUH

#include "raytracer/materials/material.cuh"

namespace rt {

__host__ __device__ inline Material create_lambertian(const Color& albedo) {
    Material mat(MaterialType::LAMBERTIAN, albedo);
    return mat;
}

__host__ __device__ inline Material create_lambertian_checker(float scale, const Color& c1, const Color& c2) {
    Material mat(MaterialType::LAMBERTIAN, Color(1, 1, 1));
    mat.tex_type = TextureType::CHECKER;
    mat.checker_tex = CheckerTexture(scale, c1, c2);
    return mat;
}

__host__ inline Material create_lambertian_noise(float scale, const Color& base_color = Color(1, 1, 1), unsigned int seed = 42) {
    Material mat(MaterialType::LAMBERTIAN, base_color);
    mat.tex_type = TextureType::NOISE;
    mat.noise_tex = NoiseTexture(scale, base_color, seed);
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
    attenuation = mat.get_albedo(rec.u, rec.v, rec.p);
    return true;
}

} // namespace rt

#endif // RAYTRACER_MATERIALS_LAMBERTIAN_CUH
