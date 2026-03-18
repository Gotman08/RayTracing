#ifndef RAYTRACER_RENDERING_TONE_MAPPING_CUH
#define RAYTRACER_RENDERING_TONE_MAPPING_CUH

/**
 * @file tone_mapping.cuh
 * @brief Fonctions de tone mapping et de correction gamma pour le rendu
 * @details Ce fichier fournit les outils de post-traitement des couleurs calculees
 *          par le raytracer. Le tone mapping compresse les valeurs HDR (qui peuvent
 *          depasser 1.0) dans l'intervalle [0, 1] via l'operateur de Reinhard.
 *          La correction gamma convertit ensuite l'espace lineaire en espace sRGB
 *          pour un affichage correct a l'ecran. Ces fonctions sont utilisables
 *          a la fois sur CPU (__host__) et GPU (__device__).
 */

#include "raytracer/core/vec3.cuh"

namespace rt {

/**
 * @brief Applique l'operateur de tone mapping de Reinhard pour compresser les couleurs HDR
 * @details En raytracing, les couleurs calculees peuvent avoir des valeurs superieures a 1.0
 *          (High Dynamic Range). L'operateur de Reinhard utilise la formule x / (1 + x) sur
 *          chaque composante RGB pour ramener doucement les valeurs dans [0, 1] sans
 *          ecreter brutalement les hautes lumieres. C'est un compromis simple et efficace.
 * @param hdr La couleur en HDR (valeurs potentiellement > 1.0)
 * @return La couleur compressees dans l'intervalle [0, 1]
 */
__host__ __device__ inline Color apply_tone_mapping(const Color& hdr) {
    return Color(
        hdr.x / (1.0f + hdr.x),
        hdr.y / (1.0f + hdr.y),
        hdr.z / (1.0f + hdr.z)
    );
}

/**
 * @brief Applique la correction gamma pour convertir l'espace lineaire en espace sRGB
 * @details Les ecrans utilisent un espace colorimetrique non-lineaire (sRGB). Sans
 *          correction gamma, l'image apparaitrait trop sombre. Cette fonction eleve
 *          chaque composante a la puissance 1/gamma (par defaut 1/2.2) pour compenser
 *          la courbe de reponse de l'ecran. Les valeurs negatives sont clampees a 0
 *          avant l'elevation en puissance pour eviter les erreurs mathematiques.
 * @param linear La couleur en espace lineaire (apres tone mapping)
 * @param gamma La valeur du gamma (2.2 par defaut pour sRGB)
 * @return La couleur corrigee, prete pour l'affichage a l'ecran
 */
__host__ __device__ inline Color gamma_correct(const Color& linear, float gamma = 2.2f) {
    float inv_gamma = 1.0f / gamma;
    return Color(
        powf(fmaxf(0.0f, linear.x), inv_gamma),
        powf(fmaxf(0.0f, linear.y), inv_gamma),
        powf(fmaxf(0.0f, linear.z), inv_gamma)
    );
}

}

#endif
