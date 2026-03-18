#ifndef RAYTRACER_MATERIALS_LAMBERTIAN_CUH
#define RAYTRACER_MATERIALS_LAMBERTIAN_CUH

/**
 * @file lambertian.cuh
 * @brief Implementation du materiau lambertien (diffus) pour le raytracer.
 * @details Ce fichier fournit les fonctions de creation et de diffusion pour
 *          les surfaces lambertiennes. Un materiau lambertien renvoie la lumiere
 *          de maniere uniforme dans toutes les directions au-dessus de la surface,
 *          ce qui produit un aspect mat (comme de la craie ou du platre).
 */

#include "raytracer/materials/material.cuh"
#include "raytracer/core/random.cuh"

namespace rt {

/**
 * @brief Cree un materiau lambertien (diffus) avec la couleur donnee.
 * @details Initialise un objet Material de type LAMBERTIAN avec l'albedo
 *          specifie. L'albedo represente la proportion de lumiere reflechie
 *          pour chaque canal de couleur (rouge, vert, bleu).
 * @param albedo La couleur de la surface diffuse (valeurs entre 0 et 1 par canal).
 * @return Un objet Material configure comme lambertien.
 */
__host__ __device__ inline Material create_lambertian(const Color& albedo) {
    Material mat(MaterialType::LAMBERTIAN, albedo);
    return mat;
}

/**
 * @brief Calcule la diffusion d'un rayon sur une surface lambertienne (version GPU).
 * @details La direction de diffusion est calculee en ajoutant un vecteur unitaire
 *          aleatoire a la normale de la surface. Si le vecteur resultant est
 *          quasi-nul (cas degenere ou le vecteur aleatoire est oppose a la normale),
 *          on utilise directement la normale comme direction de diffusion.
 *          L'attenuation est simplement l'albedo du materiau.
 * @param mat Le materiau lambertien de la surface touchee.
 * @param r_in Le rayon incident (celui qui arrive sur la surface).
 * @param rec L'enregistrement du point d'impact (position, normale, etc.).
 * @param attenuation [sortie] La couleur d'attenuation appliquee au rayon diffuse.
 * @param scattered [sortie] Le nouveau rayon diffuse genere apres le rebond.
 * @param rand_state L'etat du generateur aleatoire CUDA (curand) pour le GPU.
 * @return true toujours, car une surface lambertienne diffuse toujours le rayon.
 */
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

/**
 * @brief Calcule la diffusion d'un rayon sur une surface lambertienne (version CPU).
 * @details Meme algorithme que scatter_lambertian(), mais utilise un generateur
 *          aleatoire CPU (CPURandom) au lieu de curand. Cette version est
 *          utilisee pour le rendu sur CPU avec OpenMP.
 * @param mat Le materiau lambertien de la surface touchee.
 * @param r_in Le rayon incident.
 * @param rec L'enregistrement du point d'impact.
 * @param attenuation [sortie] La couleur d'attenuation appliquee au rayon diffuse.
 * @param scattered [sortie] Le nouveau rayon diffuse genere.
 * @param rng Le generateur de nombres aleatoires CPU.
 * @return true toujours, car la diffusion lambertienne reussit systematiquement.
 */
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
