#ifndef RAYTRACER_CORE_HIT_RECORD_CUH
#define RAYTRACER_CORE_HIT_RECORD_CUH

/**
 * @file hit_record.cuh
 * @brief Definition de la structure HitRecord pour stocker les informations d'intersection
 * @details Lorsqu'un rayon touche un objet de la scene, toutes les informations
 *          utiles sur ce point d'impact sont stockees dans un HitRecord :
 *          position, normale, materiau, coordonnees de texture, etc.
 */

#include "raytracer/core/vec3.cuh"
#include "raytracer/core/ray.cuh"

namespace rt {

class Material; /**< Declaration anticipee de la classe Material */

/**
 * @struct HitRecord
 * @brief Stocke toutes les informations relatives a une intersection rayon-objet
 * @details Cette structure est remplie par les fonctions d'intersection des objets
 *          geometriques (spheres, triangles, etc.) et est ensuite utilisee par
 *          les materiaux pour calculer la couleur et la direction du rayon rebondi.
 */
struct HitRecord {
    Point3 p;       /**< Point d'intersection dans l'espace 3D */
    Vec3 normal;    /**< Normale a la surface au point d'intersection (toujours orientee vers l'exterieur du cote du rayon) */
    Material* mat;  /**< Pointeur vers le materiau de l'objet touche */
    float t;        /**< Parametre t du rayon au point d'intersection (P = orig + t * dir) */
    float u;        /**< Coordonnee de texture u (horizontale) dans [0, 1] */
    float v;        /**< Coordonnee de texture v (verticale) dans [0, 1] */
    bool front_face; /**< true si le rayon frappe la face exterieure de l'objet */

    /**
     * @brief Determine si le rayon touche la face avant ou arriere et ajuste la normale
     * @details Compare la direction du rayon avec la normale sortante de la geometrie.
     *          Si le produit scalaire est negatif, le rayon arrive de l'exterieur (front face).
     *          La normale stockee est toujours orientee contre la direction du rayon,
     *          ce qui est necessaire pour les calculs d'eclairage et de refraction.
     * @param r Le rayon qui a touche l'objet
     * @param outward_normal La normale geometrique sortante (doit etre unitaire)
     */
    __host__ __device__ void set_face_normal(const Ray& r, const Vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

}

#endif
