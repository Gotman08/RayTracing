#ifndef RAYTRACER_CORE_RAY_CUH
#define RAYTRACER_CORE_RAY_CUH

/**
 * @file ray.cuh
 * @brief Definition de la classe Ray representant un rayon lumineux
 * @details Un rayon est l'element fondamental du ray tracing. Il est defini
 *          par un point d'origine et une direction. Un parametre de temps
 *          est aussi stocke pour gerer le flou de mouvement (motion blur).
 */

#include "raytracer/core/vec3.cuh"

namespace rt {

/**
 * @class Ray
 * @brief Represente un rayon defini par une origine, une direction et un temps
 * @details Le rayon est modelise par l'equation parametrique P(t) = orig + t * dir,
 *          ou t est un parametre reel. En faisant varier t, on obtient tous les
 *          points le long du rayon. Le champ temps permet de gerer le motion blur
 *          en associant un instant a chaque rayon lance.
 */
class Ray {
public:
    Point3 orig; /**< Point d'origine du rayon */
    Vec3 dir;    /**< Direction du rayon (pas forcement normalisee) */
    float tm;    /**< Instant auquel le rayon est lance (pour le motion blur) */

    /**
     * @brief Constructeur par defaut
     * @details Cree un rayon a l'origine avec une direction nulle et un temps de 0
     */
    __host__ __device__ Ray() : orig(), dir(), tm(0.0f) {}

    /**
     * @brief Constructeur parametrique du rayon
     * @param origin Le point d'origine du rayon
     * @param direction La direction du rayon
     * @param time L'instant du rayon (defaut : 0.0)
     */
    __host__ __device__ Ray(const Point3& origin, const Vec3& direction, float time = 0.0f)
        : orig(origin), dir(direction), tm(time) {}

    /**
     * @brief Accesseur pour l'origine du rayon
     * @return Reference constante vers le point d'origine
     */
    __host__ __device__ const Point3& origin() const { return orig; }

    /**
     * @brief Accesseur pour la direction du rayon
     * @return Reference constante vers le vecteur direction
     */
    __host__ __device__ const Vec3& direction() const { return dir; }

    /**
     * @brief Accesseur pour le temps du rayon
     * @return La valeur du temps associe au rayon
     */
    __host__ __device__ float time() const { return tm; }

    /**
     * @brief Calcule le point a la position parametrique t le long du rayon
     * @details Applique la formule P(t) = origine + t * direction.
     *          Avec t = 0 on obtient l'origine, t > 0 avance dans la direction.
     * @param t Le parametre de position le long du rayon
     * @return Le point 3D correspondant sur le rayon
     */
    __host__ __device__ Point3 at(float t) const {
        return orig + t * dir;
    }
};

}

#endif
