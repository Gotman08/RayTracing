#ifndef RAYTRACER_GEOMETRY_GEOMETRY_UTILS_CUH
#define RAYTRACER_GEOMETRY_GEOMETRY_UTILS_CUH

/**
 * @file geometry_utils.cuh
 * @brief Fonctions utilitaires pour les calculs geometriques.
 * @details Ce fichier contient des fonctions auxiliaires utilisees par les
 *          formes geometriques, notamment le calcul des coordonnees UV
 *          spheriques pour le placage de textures.
 */

#include "raytracer/core/vec3.cuh"
#include "raytracer/core/cuda_utils.cuh"

namespace rt {

/**
 * @brief Calcule les coordonnees de texture UV sur une sphere a partir d'un point sur sa surface.
 * @details Convertit un point sur la sphere unitaire en coordonnees UV (entre 0 et 1)
 *          en utilisant les coordonnees spheriques. L'angle theta (colatitude) est
 *          calcule via acos(-y) et l'angle phi (longitude) via atan2(-z, x) + PI.
 *          Les formules donnent :
 *          - u = phi / (2*PI) : coordonnee horizontale (tour complet de la sphere)
 *          - v = theta / PI : coordonnee verticale (du pole nord au pole sud)
 * @param p Point sur la sphere unitaire (vecteur normal sortant normalise).
 * @param u Coordonnee U de sortie (entre 0 et 1, direction horizontale).
 * @param v Coordonnee V de sortie (entre 0 et 1, direction verticale).
 */
__host__ __device__ inline void get_sphere_uv(const Vec3& p, float& u, float& v) {
    float theta = acosf(-p.y);
    float phi = atan2f(-p.z, p.x) + PI;
    u = phi / (2.0f * PI);
    v = theta / PI;
}

}

#endif
