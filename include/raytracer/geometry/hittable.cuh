#ifndef RAYTRACER_GEOMETRY_HITTABLE_CUH
#define RAYTRACER_GEOMETRY_HITTABLE_CUH

/** @file hittable.cuh
 *  @brief Classe de base des objets intersectables + enum type */

#include "raytracer/core/vec3.cuh"
#include "raytracer/core/ray.cuh"
#include "raytracer/core/aabb.cuh"
#include "raytracer/core/interval.cuh"
#include "raytracer/core/hit_record.cuh"

namespace rt {

/** @brief Types de geometrie (dispatch sans vtable pour CUDA) */
enum class HittableType {
    SPHERE,
    PLANE
};

/** @brief Base des objets intersectables (type + AABB) */
class Hittable {
public:
    HittableType type;
    AABB bbox;

    /** @brief Ctor par defaut (SPHERE) */
    __host__ __device__ Hittable() : type(HittableType::SPHERE) {}

    /** @brief Ctor avec type */
    __host__ __device__ Hittable(HittableType t) : type(t) {}

    /** @brief Acces a la bbox */
    __host__ __device__ const AABB& bounding_box() const { return bbox; }
};

}

#endif
