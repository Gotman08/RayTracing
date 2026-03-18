#ifndef RAYTRACER_MATERIALS_DIELECTRIC_CUH
#define RAYTRACER_MATERIALS_DIELECTRIC_CUH

/**
 * @file dielectric.cuh
 * @brief Implementation du materiau dielectrique (verre, eau) pour le raytracer.
 * @details Ce fichier fournit les fonctions de creation et de diffusion pour
 *          les materiaux dielectriques (transparents). Un dielectrique peut
 *          a la fois reflechir et refracter la lumiere selon la loi de Snell
 *          et l'approximation de Schlick pour la reflectance de Fresnel.
 *          C'est ce qui permet de simuler des objets en verre, en eau, etc.
 */

#include "raytracer/materials/material.cuh"
#include "raytracer/core/random.cuh"

namespace rt {

/**
 * @brief Cree un materiau dielectrique (transparent) avec l'indice de refraction donne.
 * @details Initialise un objet Material de type DIELECTRIC avec un albedo blanc
 *          (le verre pur ne colore pas la lumiere qui le traverse) et l'indice
 *          de refraction specifie. Par exemple, le verre a un ior d'environ 1.5
 *          et l'eau d'environ 1.33.
 * @param ior L'indice de refraction du milieu (rapport des vitesses de la lumiere).
 * @return Un objet Material configure comme dielectrique.
 */
__host__ __device__ inline Material create_dielectric(float ior) {
    Material mat(MaterialType::DIELECTRIC, Color(1, 1, 1));
    mat.ior = ior;
    return mat;
}

/**
 * @brief Calcule la reflectance selon l'approximation de Schlick.
 * @details L'approximation de Schlick est une formule simplifiee qui estime
 *          la proportion de lumiere reflechie a la surface d'un dielectrique
 *          en fonction de l'angle d'incidence. Plus l'angle est rasant (cosine
 *          proche de 0), plus la reflectance est elevee (effet bien visible
 *          quand on regarde un lac a angle rasant : l'eau devient un miroir).
 * @param cosine Le cosinus de l'angle entre le rayon incident et la normale.
 * @param ior Le rapport des indices de refraction entre les deux milieux.
 * @return La proportion de lumiere reflechie (entre 0 et 1).
 */
__host__ __device__ inline float reflectance(float cosine, float ior) {
    float r0 = (1.0f - ior) / (1.0f + ior);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * powf((1.0f - cosine), 5.0f);
}

/**
 * @brief Gere la reflexion ou la refraction d'un rayon sur un dielectrique (version GPU).
 * @details Cette fonction decide si le rayon est reflechi ou refracte en se basant
 *          sur deux criteres :
 *          1. La reflexion totale interne : si l'angle d'incidence est trop grand
 *             (selon la loi de Snell), la refraction est impossible et le rayon
 *             est entierement reflechi.
 *          2. L'approximation de Schlick : meme si la refraction est possible,
 *             une partie de la lumiere est reflechie de facon probabiliste.
 *          L'attenuation est toujours blanche car le verre pur ne colore pas la lumiere.
 * @param mat Le materiau dielectrique de la surface.
 * @param r_in Le rayon incident arrivant sur la surface.
 * @param rec L'enregistrement du point d'impact (position, normale, face avant/arriere).
 * @param attenuation [sortie] La couleur d'attenuation (toujours blanc pour un dielectrique).
 * @param scattered [sortie] Le rayon reflechi ou refracte genere.
 * @param rand_state L'etat du generateur aleatoire CUDA pour le choix reflexion/refraction.
 * @return true toujours, car un dielectrique produit toujours un rayon sortant.
 */
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

/**
 * @brief Gere la reflexion ou la refraction d'un rayon sur un dielectrique (version CPU).
 * @details Meme algorithme que scatter_dielectric(), mais utilise un generateur
 *          aleatoire CPU (CPURandom) pour le choix probabiliste entre reflexion
 *          et refraction. Cette version est utilisee pour le rendu CPU avec OpenMP.
 * @param mat Le materiau dielectrique de la surface.
 * @param r_in Le rayon incident.
 * @param rec L'enregistrement du point d'impact.
 * @param attenuation [sortie] La couleur d'attenuation (toujours blanc).
 * @param scattered [sortie] Le rayon reflechi ou refracte genere.
 * @param rng Le generateur de nombres aleatoires CPU.
 * @return true toujours, car un dielectrique produit toujours un rayon sortant.
 */
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
