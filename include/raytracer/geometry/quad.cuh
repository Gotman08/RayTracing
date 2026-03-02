#ifndef RAYTRACER_GEOMETRY_QUAD_CUH
#define RAYTRACER_GEOMETRY_QUAD_CUH

#include "raytracer/geometry/hittable.cuh"
#include "raytracer/core/cuda_utils.cuh"

namespace rt {

class Quad : public Hittable {
public:
    Point3 Q;       // Starting corner
    Vec3 u, v;      // Two edge vectors
    Vec3 normal;
    Vec3 w;         // Plane constant
    float D;
    Material* mat;

    __host__ __device__ Quad() : Hittable(HittableType::QUAD), mat(nullptr) {}

    __host__ __device__ Quad(const Point3& _Q, const Vec3& _u, const Vec3& _v, Material* m)
        : Hittable(HittableType::QUAD), Q(_Q), u(_u), v(_v), mat(m) {
        Vec3 n = cross(u, v);
        normal = unit_vector(n);
        D = dot(normal, Q);
        w = n / dot(n, n);

        set_bounding_box();
    }

    __host__ __device__ void set_bounding_box() {
        AABB bbox_diag1(Q, Q + u + v);
        AABB bbox_diag2(Q + u, Q + v);
        bbox = AABB(bbox_diag1, bbox_diag2);
    }

    __host__ __device__ bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const {
        float denom = dot(normal, r.direction());

        // Ray parallel to plane
        if (fabsf(denom) < EPSILON)
            return false;

        float t = (D - dot(normal, r.origin())) / denom;
        if (!ray_t.surrounds(t))
            return false;

        Point3 intersection = r.at(t);
        Vec3 planar_hitpt = intersection - Q;
        float alpha = dot(w, cross(planar_hitpt, v));
        float beta = dot(w, cross(u, planar_hitpt));

        if (!is_interior(alpha, beta, rec))
            return false;

        rec.t = t;
        rec.p = intersection;
        rec.mat = mat;
        rec.set_face_normal(r, normal);

        return true;
    }

private:
    __host__ __device__ bool is_interior(float a, float b, HitRecord& rec) const {
        Interval unit_interval(0.0f, 1.0f);

        if (!unit_interval.contains(a) || !unit_interval.contains(b))
            return false;

        rec.u = a;
        rec.v = b;
        return true;
    }
};

} // namespace rt

#endif // RAYTRACER_GEOMETRY_QUAD_CUH
