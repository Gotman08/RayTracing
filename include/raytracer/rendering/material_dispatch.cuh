#ifndef RAYTRACER_RENDERING_MATERIAL_DISPATCH_CUH
#define RAYTRACER_RENDERING_MATERIAL_DISPATCH_CUH

/**
 * @file material_dispatch.cuh
 * @brief Dispatch de l'interaction rayon-materiau selon le type de materiau
 * @details Ce fichier contient les fonctions de dispatch qui redirigent l'appel
 *          de diffusion (scatter) vers la bonne implementation selon le type du
 *          materiau (lambertien, metal ou dielectrique). Il existe une version
 *          GPU (scatter) et une version CPU (scatter_cpu) qui utilisent des
 *          generateurs aleatoires differents.
 */

#include "raytracer/core/vec3.cuh"
#include "raytracer/core/ray.cuh"
#include "raytracer/core/hit_record.cuh"
#include "raytracer/materials/material.cuh"
#include "raytracer/materials/lambertian.cuh"
#include "raytracer/materials/metal.cuh"
#include "raytracer/materials/dielectric.cuh"

namespace rt {

/**
 * @brief Dispatche l'interaction rayon-materiau selon le type de materiau (version GPU)
 * @details Cette fonction examine le type du materiau touche par le rayon et appelle
 *          la fonction de diffusion appropriee : lambertien (diffusion aleatoire),
 *          metal (reflexion avec flou) ou dielectrique (refraction/reflexion type verre).
 *          Elle utilise un switch sur l'enum MaterialType.
 * @param mat Le materiau de la surface touchee
 * @param r_in Le rayon incident
 * @param rec L'enregistrement du point d'intersection
 * @param attenuation La couleur d'attenuation du materiau (parametre de sortie)
 * @param scattered Le rayon diffuse produit par l'interaction (parametre de sortie)
 * @param rand_state L'etat du generateur aleatoire curand
 * @return true si le rayon a ete diffuse, false s'il a ete absorbe
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
        default:
            return false;
    }
}

/**
 * @brief Dispatche l'interaction rayon-materiau selon le type de materiau (version CPU)
 * @details Version CPU de scatter(). Meme logique de dispatch par switch, mais utilise
 *          les variantes CPU des fonctions de diffusion (scatter_lambertian_cpu, etc.)
 *          qui prennent un CPURandom au lieu d'un curandState.
 * @param mat Le materiau de la surface touchee
 * @param r_in Le rayon incident
 * @param rec L'enregistrement du point d'intersection
 * @param attenuation La couleur d'attenuation du materiau (parametre de sortie)
 * @param scattered Le rayon diffuse produit par l'interaction (parametre de sortie)
 * @param rng Le generateur de nombres aleatoires CPU
 * @return true si le rayon a ete diffuse, false s'il a ete absorbe
 */
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
