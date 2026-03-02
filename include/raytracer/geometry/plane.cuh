#ifndef RAYTRACER_GEOMETRY_PLANE_CUH
#define RAYTRACER_GEOMETRY_PLANE_CUH

#include "raytracer/geometry/hittable.cuh"
#include "raytracer/core/cuda_utils.cuh"

namespace rt {

class Plane : public Hittable {
public:
    Point3 point;
    Vec3 normal;
    Material* mat;

    __host__ __device__ Plane() : Hittable(HittableType::PLANE), mat(nullptr) {}

    __host__ __device__ Plane(const Point3& p, const Vec3& n, Material* m)
        : Hittable(HittableType::PLANE), point(p), normal(unit_vector(n)), mat(m) {
        // Infinite plane has infinite bounding box in two directions
        // Use a large but finite box for practical purposes
        float big = 1e10f;
        Vec3 tangent1, tangent2;

        if (fabsf(normal.x) > 0.9f) {
            tangent1 = cross(normal, Vec3(0, 1, 0)).normalized();
        } else {
            tangent1 = cross(normal, Vec3(1, 0, 0)).normalized();
        }
        tangent2 = cross(normal, tangent1);

        Point3 min_p = point - big * tangent1 - big * tangent2 - EPSILON * normal;
        Point3 max_p = point + big * tangent1 + big * tangent2 + EPSILON * normal;
        bbox = AABB(min_p, max_p);
    }

    __host__ __device__ bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const {
        float denom = dot(normal, r.direction());

        // Ray parallel to plane
        if (fabsf(denom) < EPSILON)
            return false;

        float t = dot(point - r.origin(), normal) / denom;

        if (!ray_t.surrounds(t))
            return false;

        rec.t = t;
        rec.p = r.at(t);
        rec.set_face_normal(r, normal);
        rec.mat = mat;

        // UV coordinates based on projection
        Vec3 local = rec.p - point;
        Vec3 tangent1, tangent2;
        if (fabsf(normal.x) > 0.9f) {
            tangent1 = cross(normal, Vec3(0, 1, 0)).normalized();
        } else {
            tangent1 = cross(normal, Vec3(1, 0, 0)).normalized();
        }
        tangent2 = cross(normal, tangent1);

        rec.u = dot(local, tangent1) * 0.1f;
        rec.v = dot(local, tangent2) * 0.1f;
        rec.u = rec.u - floorf(rec.u);
        rec.v = rec.v - floorf(rec.v);

        return true;
    }
};

} // namespace rt

#endif // RAYTRACER_GEOMETRY_PLANE_CUH
