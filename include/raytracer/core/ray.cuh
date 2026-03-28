#ifndef RAYTRACER_CORE_RAY_CUH
#define RAYTRACER_CORE_RAY_CUH

/** @file ray.cuh
 *  @brief Classe Ray - rayon parametrique P(t) = orig + t*dir */

#include "raytracer/core/vec3.cuh"

namespace rt {

/** @brief Rayon = origine + direction + temps (motion blur)
 *  Aligne a 16 octets pour acces memoire coalescents */
class __align__(16) Ray {
public:
    Point3 orig;
    Vec3 dir;
    float tm;    ///< Instant du rayon (motion blur)

    /** @brief Ctor par defaut */
    __host__ __device__ Ray() : orig(), dir(), tm(0.0f) {}

    /** @brief Ctor complet (origine, dir, temps) */
    __host__ __device__ Ray(const Point3& origin, const Vec3& direction, float time = 0.0f)
        : orig(origin), dir(direction), tm(time) {}

    __host__ __device__ const Point3& origin() const { return orig; }
    __host__ __device__ const Vec3& direction() const { return dir; }
    __host__ __device__ float time() const { return tm; }

    /** @brief Point a la position parametrique t */
    __host__ __device__ Point3 at(float t) const {
        return orig + t * dir;
    }
};

}

#endif
