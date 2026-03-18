#ifndef RAYTRACER_MATERIALS_METAL_CUH
#define RAYTRACER_MATERIALS_METAL_CUH

/**
 * @file metal.cuh
 * @brief Implementation du materiau metallique pour le raytracer.
 * @details Ce fichier fournit les fonctions de creation et de reflexion pour
 *          les surfaces metalliques. Un metal reflechit les rayons comme un miroir,
 *          avec un parametre de flou (fuzz) qui permet de simuler des surfaces
 *          metalliques plus ou moins polies (miroir parfait vs metal brosse).
 */

#include "raytracer/materials/material.cuh"
#include "raytracer/core/random.cuh"

namespace rt {

/**
 * @brief Cree un materiau metallique avec la couleur et le flou donnes.
 * @details Initialise un objet Material de type METAL. Le parametre fuzz
 *          est borne a 1.0 maximum : a 0 le metal est un miroir parfait,
 *          a 1 la reflexion est tres perturbee (surface rugueuse).
 * @param albedo La couleur du metal (teinte de la reflexion).
 * @param fuzz Le facteur de flou de la reflexion (0 = miroir, 1 = tres flou).
 * @return Un objet Material configure comme metallique.
 */
__host__ __device__ inline Material create_metal(const Color& albedo, float fuzz) {
    Material mat(MaterialType::METAL, albedo);
    mat.fuzz = fuzz < 1.0f ? fuzz : 1.0f;
    return mat;
}

/**
 * @brief Calcule la reflexion d'un rayon sur une surface metallique (version GPU).
 * @details Le rayon incident est reflechi par rapport a la normale de la surface.
 *          Le vecteur reflechi est ensuite perturbe par un vecteur aleatoire
 *          pondere par le facteur de flou (fuzz), ce qui simule l'imperfection
 *          de la surface. Si le rayon reflechi part sous la surface (produit
 *          scalaire negatif avec la normale), la reflexion est rejetee.
 * @param mat Le materiau metallique de la surface.
 * @param r_in Le rayon incident arrivant sur la surface.
 * @param rec L'enregistrement du point d'impact (position, normale, etc.).
 * @param attenuation [sortie] La couleur d'attenuation (albedo du metal).
 * @param scattered [sortie] Le rayon reflechi genere.
 * @param rand_state L'etat du generateur aleatoire CUDA pour la perturbation.
 * @return true si le rayon reflechi est du bon cote de la surface, false sinon.
 */
__device__ inline bool scatter_metal(
    const Material& mat,
    const Ray& r_in,
    const HitRecord& rec,
    Color& attenuation,
    Ray& scattered,
    curandState* rand_state
) {
    Vec3 reflected = reflect(r_in.direction(), rec.normal);
    reflected = unit_vector(reflected) + (mat.fuzz * random_unit_vector(rand_state));
    scattered = Ray(rec.p, reflected, r_in.time());
    attenuation = mat.albedo;
    return (dot(scattered.direction(), rec.normal) > 0);
}

/**
 * @brief Calcule la reflexion d'un rayon sur une surface metallique (version CPU).
 * @details Meme algorithme que scatter_metal(), mais utilise un generateur
 *          aleatoire CPU (CPURandom) pour la perturbation du vecteur reflechi.
 *          Cette version est utilisee lors du rendu CPU avec OpenMP.
 * @param mat Le materiau metallique de la surface.
 * @param r_in Le rayon incident.
 * @param rec L'enregistrement du point d'impact.
 * @param attenuation [sortie] La couleur d'attenuation (albedo du metal).
 * @param scattered [sortie] Le rayon reflechi genere.
 * @param rng Le generateur de nombres aleatoires CPU.
 * @return true si le rayon reflechi est du bon cote de la surface, false sinon.
 */
inline bool scatter_metal_cpu(
    const Material& mat,
    const Ray& r_in,
    const HitRecord& rec,
    Color& attenuation,
    Ray& scattered,
    CPURandom& rng
) {
    Vec3 reflected = reflect(r_in.direction(), rec.normal);
    reflected = unit_vector(reflected) + (mat.fuzz * random_unit_vector(rng));
    scattered = Ray(rec.p, reflected, r_in.time());
    attenuation = mat.albedo;
    return (dot(scattered.direction(), rec.normal) > 0);
}

}

#endif
