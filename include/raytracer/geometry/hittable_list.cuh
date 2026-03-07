#ifndef RAYTRACER_GEOMETRY_HITTABLE_LIST_CUH
#define RAYTRACER_GEOMETRY_HITTABLE_LIST_CUH

#include "raytracer/geometry/hittable.cuh"
#include "raytracer/geometry/sphere.cuh"
#include "raytracer/geometry/plane.cuh"

namespace rt {

union HittableData {
    Sphere sphere;
    Plane plane;

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
            case HittableType::PLANE:
                return data.plane.hit(r, ray_t, rec);
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

    __host__ __device__ void add_plane(const Point3& point, const Vec3& normal, Material* mat) {
        if (count >= capacity) return;
        objects[count].type = HittableType::PLANE;
        objects[count].data.plane = Plane(point, normal, mat);
        objects[count].bbox = objects[count].data.plane.bounding_box();
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

}

#endif
