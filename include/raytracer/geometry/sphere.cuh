#ifndef RAYTRACER_GEOMETRY_SPHERE_CUH
#define RAYTRACER_GEOMETRY_SPHERE_CUH

/** @file sphere.cuh
 *  @brief Sphere 3D : intersection par eq. quadratique */

#include "raytracer/geometry/hittable.cuh"
#include "raytracer/geometry/geometry_utils.cuh"
#include "raytracer/core/cuda_utils.cuh"

namespace rt {

/** @brief Sphere : centre + rayon + materiau */
class Sphere : public Hittable {
public:
    Point3 center;
    float radius;
    Material* mat;

    /** @brief Ctor par defaut, rayon 0 */
    __host__ __device__ Sphere() : Hittable(HittableType::SPHERE), radius(0), mat(nullptr) {}

    /** @brief Ctor : init centre/rayon/mat + calcul AABB */
    __host__ __device__ Sphere(const Point3& c, float r, Material* m)
        : Hittable(HittableType::SPHERE), center(c), radius(fmaxf(0.0f, r)), mat(m) {
        Vec3 rvec(radius, radius, radius);
        bbox = AABB(center - rvec, center + rvec);
    }

    /** @brief Intersection rayon-sphere, eq. quadratique + UV */
    __host__ __device__ bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const {
        Vec3 oc = center - r.origin();
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
        Vec3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);
        get_sphere_uv(outward_normal, rec.u, rec.v);
        rec.mat = mat;

        return true;
    }
};

}

#endif
