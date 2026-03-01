#ifndef RAYTRACER_CORE_HIT_RECORD_CUH
#define RAYTRACER_CORE_HIT_RECORD_CUH

#include "raytracer/core/vec3.cuh"
#include "raytracer/core/ray.cuh"

namespace rt {

// Forward declaration to break circular dependency
class Material;

struct HitRecord {
    Point3 p;
    Vec3 normal;
    Material* mat;
    float t;
    float u;
    float v;
    bool front_face;

    __host__ __device__ void set_face_normal(const Ray& r, const Vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

} // namespace rt

#endif // RAYTRACER_CORE_HIT_RECORD_CUH
