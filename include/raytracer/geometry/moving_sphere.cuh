#ifndef RAYTRACER_GEOMETRY_MOVING_SPHERE_CUH
#define RAYTRACER_GEOMETRY_MOVING_SPHERE_CUH

#include "raytracer/geometry/hittable.cuh"
#include "raytracer/geometry/geometry_utils.cuh"
#include "raytracer/core/cuda_utils.cuh"

namespace rt {

class MovingSphere : public Hittable {
public:
    Point3 center0, center1;
    float radius;
    Material* mat;

    __host__ __device__ MovingSphere()
        : Hittable(HittableType::MOVING_SPHERE), radius(0), mat(nullptr) {}

    __host__ __device__ MovingSphere(const Point3& c0, const Point3& c1, float r, Material* m)
        : Hittable(HittableType::MOVING_SPHERE), center0(c0), center1(c1),
          radius(fmaxf(0.0f, r)), mat(m) {
        // Bounding box encompasses the sphere at both t=0 and t=1
        Vec3 rvec(radius, radius, radius);
        AABB box0(center0 - rvec, center0 + rvec);
        AABB box1(center1 - rvec, center1 + rvec);
        bbox = AABB(box0, box1);
    }

    __host__ __device__ Point3 sphere_center(float time) const {
        return center0 + time * (center1 - center0);
    }

    __host__ __device__ bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const {
        Point3 current_center = sphere_center(r.time());
        Vec3 oc = current_center - r.origin();

        float a = r.direction().length_squared();
        float h = dot(r.direction(), oc);
        float c = oc.length_squared() - radius * radius;
        float discriminant = h * h - a * c;

        if (discriminant < 0)
            return false;

        float sqrtd = sqrtf(discriminant);

        float root = (h - sqrtd) / a;
        if (!ray_t.surrounds(root)) {
            root = (h + sqrtd) / a;
            if (!ray_t.surrounds(root))
                return false;
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        Vec3 outward_normal = (rec.p - current_center) / radius;
        rec.set_face_normal(r, outward_normal);
        get_sphere_uv(outward_normal, rec.u, rec.v);
        rec.mat = mat;

        return true;
    }
};

} // namespace rt

#endif // RAYTRACER_GEOMETRY_MOVING_SPHERE_CUH
