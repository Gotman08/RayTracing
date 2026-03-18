#ifndef RAYTRACER_GEOMETRY_HITTABLE_CUH
#define RAYTRACER_GEOMETRY_HITTABLE_CUH

/**
 * @file hittable.cuh
 * @brief Classe de base abstraite pour tous les objets intersectables de la scene.
 * @details Ce fichier definit l'enumeration des types d'objets disponibles ainsi que
 *          la classe de base Hittable dont heritent les formes geometriques concretes
 *          (sphere, plan, etc.). Chaque objet possede un type et une boite englobante (AABB).
 */

#include "raytracer/core/vec3.cuh"
#include "raytracer/core/ray.cuh"
#include "raytracer/core/aabb.cuh"
#include "raytracer/core/interval.cuh"
#include "raytracer/core/hit_record.cuh"

namespace rt {

/**
 * @enum HittableType
 * @brief Enumeration des differents types d'objets geometriques supportes.
 * @details On utilise cette enumeration pour identifier le type concret d'un objet
 *          sans avoir recours au polymorphisme dynamique (incompatible avec CUDA).
 *          Cela permet de faire un switch sur le type dans le code GPU.
 */
enum class HittableType {
    SPHERE, ///< Objet de type sphere
    PLANE   ///< Objet de type plan infini
};

/**
 * @class Hittable
 * @brief Classe de base pour tous les objets de la scene pouvant etre intersectes par un rayon.
 * @details Cette classe stocke le type de l'objet et sa boite englobante (AABB).
 *          Les classes derivees (Sphere, Plane) heritent de Hittable et implementent
 *          leur propre methode hit() pour calculer l'intersection avec un rayon.
 *          Compatible host (CPU) et device (GPU) grace aux qualificateurs CUDA.
 */
class Hittable {
public:
    HittableType type; ///< Type de l'objet (SPHERE ou PLANE)
    AABB bbox;         ///< Boite englobante alignee sur les axes (Axis-Aligned Bounding Box)

    /**
     * @brief Constructeur par defaut.
     * @details Initialise l'objet comme une sphere par defaut.
     */
    __host__ __device__ Hittable() : type(HittableType::SPHERE) {}

    /**
     * @brief Constructeur avec type specifie.
     * @param t Le type de l'objet geometrique a creer.
     */
    __host__ __device__ Hittable(HittableType t) : type(t) {}

    /**
     * @brief Retourne la boite englobante de l'objet.
     * @return Reference constante vers l'AABB de l'objet.
     */
    __host__ __device__ const AABB& bounding_box() const { return bbox; }
};

}

#endif
