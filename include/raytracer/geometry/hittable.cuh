#ifndef RAYTRACER_GEOMETRY_HITTABLE_CUH
#define RAYTRACER_GEOMETRY_HITTABLE_CUH

#include "raytracer/core/vec3.cuh"
#include "raytracer/core/ray.cuh"
#include "raytracer/core/aabb.cuh"
#include "raytracer/core/interval.cuh"

namespace rt {

// Forward declarations
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

// Object types for GPU dispatch (avoiding virtual functions)
enum class HittableType {
    SPHERE,
    MOVING_SPHERE,
    PLANE,
    QUAD,
    TRIANGLE,
    BOX,
    BVH_NODE,
    HITTABLE_LIST
};

class Hittable {
public:
    HittableType type;
    AABB bbox;

    __host__ __device__ Hittable() : type(HittableType::SPHERE) {}
    __host__ __device__ Hittable(HittableType t) : type(t) {}

    __host__ __device__ const AABB& bounding_box() const { return bbox; }
};

} // namespace rt

#endif // RAYTRACER_GEOMETRY_HITTABLE_CUH
