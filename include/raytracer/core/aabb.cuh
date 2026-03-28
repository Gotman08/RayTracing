#ifndef RAYTRACER_CORE_AABB_CUH
#define RAYTRACER_CORE_AABB_CUH

/** @file aabb.cuh
 *  @brief AABB - boite englobante alignee aux axes pour le BVH */

#include "raytracer/core/vec3.cuh"
#include "raytracer/core/ray.cuh"
#include "raytracer/core/interval.cuh"

namespace rt {

/** @brief Boite englobante axis-aligned (3 intervalles x,y,z) */
class AABB {
public:
    Interval x, y, z;

    /** @brief Ctor par defaut -> AABB vide */
    __host__ __device__ AABB() {}

    /** @brief Ctor depuis 3 intervalles (avec padding auto) */
    __host__ __device__ AABB(const Interval& x, const Interval& y, const Interval& z)
        : x(x), y(y), z(z) {
        pad_to_minimums();
    }

    /** @brief Ctor depuis 2 coins opposes */
    __host__ __device__ AABB(const Point3& a, const Point3& b) {
        x = (a.x <= b.x) ? Interval(a.x, b.x) : Interval(b.x, a.x);
        y = (a.y <= b.y) ? Interval(a.y, b.y) : Interval(b.y, a.y);
        z = (a.z <= b.z) ? Interval(a.z, b.z) : Interval(b.z, a.z);
        pad_to_minimums();
    }

    /** @brief Union de deux AABB */
    __host__ __device__ AABB(const AABB& box0, const AABB& box1) {
        x = Interval(box0.x, box1.x);
        y = Interval(box0.y, box1.y);
        z = Interval(box0.z, box1.z);
    }

    /** @brief Intervalle de l'axe n (0=x, 1=y, 2=z) */
    __host__ __device__ const Interval& axis_interval(int n) const {
        if (n == 1) return y;
        if (n == 2) return z;
        return x;
    }

    /** @brief Intersection rayon-boite (slab method) */
    __host__ __device__ bool hit(const Ray& r, Interval ray_t) const {
        const Point3& orig = r.origin();
        const Vec3& dir = r.direction();

        float tmin = ray_t.min;
        float tmax = ray_t.max;

        {
            float invD = 1.0f / dir.x;
            float t0 = (x.min - orig.x) * invD;
            float t1 = (x.max - orig.x) * invD;
            if (invD < 0.0f) { float tmp = t0; t0 = t1; t1 = tmp; }
            tmin = t0 > tmin ? t0 : tmin;
            tmax = t1 < tmax ? t1 : tmax;
            if (tmax <= tmin) return false;
        }

        {
            float invD = 1.0f / dir.y;
            float t0 = (y.min - orig.y) * invD;
            float t1 = (y.max - orig.y) * invD;
            if (invD < 0.0f) { float tmp = t0; t0 = t1; t1 = tmp; }
            tmin = t0 > tmin ? t0 : tmin;
            tmax = t1 < tmax ? t1 : tmax;
            if (tmax <= tmin) return false;
        }

        {
            float invD = 1.0f / dir.z;
            float t0 = (z.min - orig.z) * invD;
            float t1 = (z.max - orig.z) * invD;
            if (invD < 0.0f) { float tmp = t0; t0 = t1; t1 = tmp; }
            tmin = t0 > tmin ? t0 : tmin;
            tmax = t1 < tmax ? t1 : tmax;
            if (tmax <= tmin) return false;
        }

        return true;
    }

    /** @brief Axe le plus long (0=x, 1=y, 2=z) */
    __host__ __device__ int longest_axis() const {
        if (x.size() > y.size())
            return x.size() > z.size() ? 0 : 2;
        else
            return y.size() > z.size() ? 1 : 2;
    }

    /** @brief Centroide de la boite */
    __host__ __device__ Point3 centroid() const {
        return Point3(
            (x.min + x.max) * 0.5f,
            (y.min + y.max) * 0.5f,
            (z.min + z.max) * 0.5f
        );
    }

    /** @brief Aire de surface (pour heuristique SAH) */
    __host__ __device__ float surface_area() const {
        float dx = x.size();
        float dy = y.size();
        float dz = z.size();
        return 2.0f * (dx * dy + dy * dz + dz * dx);
    }

private:
    /** @brief Padding min pour eviter les axes degeneres */
    __host__ __device__ void pad_to_minimums() {
        float delta = 0.0001f;
        if (x.size() < delta) x = x.expand(delta);
        if (y.size() < delta) y = y.expand(delta);
        if (z.size() < delta) z = z.expand(delta);
    }
};

}

#endif
