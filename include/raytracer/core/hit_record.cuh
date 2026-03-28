#ifndef RAYTRACER_CORE_HIT_RECORD_CUH
#define RAYTRACER_CORE_HIT_RECORD_CUH

/** @file hit_record.cuh
 *  @brief HitRecord - infos d'intersection rayon-objet */

#include "raytracer/core/vec3.cuh"
#include "raytracer/core/ray.cuh"

namespace rt {

class Material;

/** @brief Donnees d'un point d'impact rayon-surface */
struct HitRecord {
    Point3 p;
    Vec3 normal;
    Material* mat;
    float t;
    float u, v;      ///< Coords de texture
    bool front_face;

    /** @brief Oriente la normale du cote du rayon incident */
    __host__ __device__ void set_face_normal(const Ray& r, const Vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

}

#endif
