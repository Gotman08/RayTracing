#ifndef RAYTRACER_GEOMETRY_TRIANGLE_CUH
#define RAYTRACER_GEOMETRY_TRIANGLE_CUH

#include "raytracer/geometry/hittable.cuh"
#include "raytracer/core/cuda_utils.cuh"

namespace rt {

class Triangle : public Hittable {
public:
    Point3 v0, v1, v2;
    Vec3 n0, n1, n2;
    Material* mat;
    bool smooth_shading;

    __host__ __device__ Triangle()
        : Hittable(HittableType::TRIANGLE), mat(nullptr), smooth_shading(false) {}

    __host__ __device__ Triangle(const Point3& a, const Point3& b, const Point3& c, Material* m)
        : Hittable(HittableType::TRIANGLE), v0(a), v1(b), v2(c), mat(m), smooth_shading(false) {
        Vec3 edge1 = v1 - v0;
        Vec3 edge2 = v2 - v0;
        Vec3 flat_normal = unit_vector(cross(edge1, edge2));
        n0 = n1 = n2 = flat_normal;
        compute_bounding_box();
    }

    __host__ __device__ void set_normals(const Vec3& na, const Vec3& nb, const Vec3& nc) {
        n0 = na;
        n1 = nb;
        n2 = nc;
        smooth_shading = true;
    }

    __host__ __device__ void compute_bounding_box() {
        Point3 min_p(
            fminf(fminf(v0.x, v1.x), v2.x) - EPSILON,
            fminf(fminf(v0.y, v1.y), v2.y) - EPSILON,
            fminf(fminf(v0.z, v1.z), v2.z) - EPSILON
        );
        Point3 max_p(
            fmaxf(fmaxf(v0.x, v1.x), v2.x) + EPSILON,
            fmaxf(fmaxf(v0.y, v1.y), v2.y) + EPSILON,
            fmaxf(fmaxf(v0.z, v1.z), v2.z) + EPSILON
        );
        bbox = AABB(min_p, max_p);
    }

    // Moller-Trumbore intersection algorithm
    __host__ __device__ bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const {
        Vec3 edge1 = v1 - v0;
        Vec3 edge2 = v2 - v0;
        Vec3 h = cross(r.direction(), edge2);
        float a = dot(edge1, h);

        if (fabsf(a) < EPSILON)
            return false;

        float f = 1.0f / a;
        Vec3 s = r.origin() - v0;
        float u = f * dot(s, h);

        if (u < 0.0f || u > 1.0f)
            return false;

        Vec3 q = cross(s, edge1);
        float v = f * dot(r.direction(), q);

        if (v < 0.0f || u + v > 1.0f)
            return false;

        float t = f * dot(edge2, q);

        if (!ray_t.surrounds(t))
            return false;

        rec.t = t;
        rec.p = r.at(t);
        rec.u = u;
        rec.v = v;
        rec.mat = mat;

        Vec3 outward_normal;
        if (smooth_shading) {
            float w = 1.0f - u - v;
            outward_normal = unit_vector(w * n0 + u * n1 + v * n2);
        } else {
            outward_normal = unit_vector(cross(edge1, edge2));
        }
        rec.set_face_normal(r, outward_normal);

        return true;
    }
};

} // namespace rt

#endif // RAYTRACER_GEOMETRY_TRIANGLE_CUH
