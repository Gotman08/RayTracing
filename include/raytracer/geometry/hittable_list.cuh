#ifndef RAYTRACER_GEOMETRY_HITTABLE_LIST_CUH
#define RAYTRACER_GEOMETRY_HITTABLE_LIST_CUH

#include "raytracer/geometry/hittable.cuh"
#include "raytracer/geometry/sphere.cuh"
#include "raytracer/geometry/moving_sphere.cuh"
#include "raytracer/geometry/plane.cuh"
#include "raytracer/geometry/quad.cuh"
#include "raytracer/geometry/triangle.cuh"
#include "raytracer/geometry/box.cuh"

namespace rt {

// Union to store different object types
union HittableData {
    Sphere sphere;
    MovingSphere moving_sphere;
    Plane plane;
    Quad quad;
    Triangle triangle;
    Box box;

    __host__ __device__ HittableData() {}
};

struct HittableObject {
    HittableType type;
    HittableData data;
    AABB bbox;

    __host__ __device__ HittableObject() : type(HittableType::SPHERE) {}

    __host__ __device__ bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const {
        switch (type) {
            case HittableType::SPHERE:
                return data.sphere.hit(r, ray_t, rec);
            case HittableType::MOVING_SPHERE:
                return data.moving_sphere.hit(r, ray_t, rec);
            case HittableType::PLANE:
                return data.plane.hit(r, ray_t, rec);
            case HittableType::QUAD:
                return data.quad.hit(r, ray_t, rec);
            case HittableType::TRIANGLE:
                return data.triangle.hit(r, ray_t, rec);
            case HittableType::BOX:
                return data.box.hit(r, ray_t, rec);
            default:
                return false;
        }
    }
};

class HittableList {
public:
    HittableObject* objects;
    int count;
    int capacity;
    AABB bbox;

    __host__ __device__ HittableList() : objects(nullptr), count(0), capacity(0) {}

    __host__ __device__ void add_sphere(const Point3& center, float radius, Material* mat) {
        if (count >= capacity) return;
        objects[count].type = HittableType::SPHERE;
        objects[count].data.sphere = Sphere(center, radius, mat);
        objects[count].bbox = objects[count].data.sphere.bounding_box();
        bbox = (count == 0) ? objects[count].bbox : AABB(bbox, objects[count].bbox);
        count++;
    }

    __host__ __device__ void add_moving_sphere(const Point3& c0, const Point3& c1, float radius, Material* mat) {
        if (count >= capacity) return;
        objects[count].type = HittableType::MOVING_SPHERE;
        objects[count].data.moving_sphere = MovingSphere(c0, c1, radius, mat);
        objects[count].bbox = objects[count].data.moving_sphere.bounding_box();
        bbox = (count == 0) ? objects[count].bbox : AABB(bbox, objects[count].bbox);
        count++;
    }

    __host__ __device__ void add_quad(const Point3& Q, const Vec3& u, const Vec3& v, Material* mat) {
        if (count >= capacity) return;
        objects[count].type = HittableType::QUAD;
        objects[count].data.quad = Quad(Q, u, v, mat);
        objects[count].bbox = objects[count].data.quad.bounding_box();
        bbox = (count == 0) ? objects[count].bbox : AABB(bbox, objects[count].bbox);
        count++;
    }

    __host__ __device__ void add_box(const Point3& a, const Point3& b, Material* mat) {
        if (count >= capacity) return;
        objects[count].type = HittableType::BOX;
        objects[count].data.box = Box(a, b, mat);
        objects[count].bbox = objects[count].data.box.bounding_box();
        bbox = (count == 0) ? objects[count].bbox : AABB(bbox, objects[count].bbox);
        count++;
    }

    __host__ __device__ bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const {
        HitRecord temp_rec;
        bool hit_anything = false;
        float closest_so_far = ray_t.max;

        for (int i = 0; i < count; i++) {
            if (objects[i].hit(r, Interval(ray_t.min, closest_so_far), temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }

    __host__ __device__ const AABB& bounding_box() const { return bbox; }
};

} // namespace rt

#endif // RAYTRACER_GEOMETRY_HITTABLE_LIST_CUH
