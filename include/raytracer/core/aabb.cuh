#ifndef RAYTRACER_CORE_AABB_CUH
#define RAYTRACER_CORE_AABB_CUH

#include "raytracer/core/vec3.cuh"
#include "raytracer/core/ray.cuh"
#include "raytracer/core/interval.cuh"

namespace rt {

class AABB {
public:
    Interval x, y, z;

    __host__ __device__ AABB() {}

    __host__ __device__ AABB(const Interval& x, const Interval& y, const Interval& z)
        : x(x), y(y), z(z) {
        pad_to_minimums();
    }

    __host__ __device__ AABB(const Point3& a, const Point3& b) {
        x = (a.x <= b.x) ? Interval(a.x, b.x) : Interval(b.x, a.x);
        y = (a.y <= b.y) ? Interval(a.y, b.y) : Interval(b.y, a.y);
        z = (a.z <= b.z) ? Interval(a.z, b.z) : Interval(b.z, a.z);
        pad_to_minimums();
    }

    __host__ __device__ AABB(const AABB& box0, const AABB& box1) {
        x = Interval(box0.x, box1.x);
        y = Interval(box0.y, box1.y);
        z = Interval(box0.z, box1.z);
    }

    __host__ __device__ const Interval& axis_interval(int n) const {
        if (n == 1) return y;
        if (n == 2) return z;
        return x;
    }

    // OPTIMIZED: Unrolled, branchless AABB intersection
    __host__ __device__ bool hit(const Ray& r, Interval ray_t) const {
        const Point3& orig = r.origin();
        const Vec3& dir = r.direction();

        float tmin = ray_t.min;
        float tmax = ray_t.max;

        // X axis
        {
            float invD = 1.0f / dir.x;
            float t0 = (x.min - orig.x) * invD;
            float t1 = (x.max - orig.x) * invD;
            if (invD < 0.0f) { float tmp = t0; t0 = t1; t1 = tmp; }
            tmin = t0 > tmin ? t0 : tmin;
            tmax = t1 < tmax ? t1 : tmax;
            if (tmax <= tmin) return false;
        }

        // Y axis
        {
            float invD = 1.0f / dir.y;
            float t0 = (y.min - orig.y) * invD;
            float t1 = (y.max - orig.y) * invD;
            if (invD < 0.0f) { float tmp = t0; t0 = t1; t1 = tmp; }
            tmin = t0 > tmin ? t0 : tmin;
            tmax = t1 < tmax ? t1 : tmax;
            if (tmax <= tmin) return false;
        }

        // Z axis
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

    __host__ __device__ int longest_axis() const {
        if (x.size() > y.size())
            return x.size() > z.size() ? 0 : 2;
        else
            return y.size() > z.size() ? 1 : 2;
    }

    __host__ __device__ Point3 centroid() const {
        return Point3(
            (x.min + x.max) * 0.5f,
            (y.min + y.max) * 0.5f,
            (z.min + z.max) * 0.5f
        );
    }

    __host__ __device__ float surface_area() const {
        float dx = x.size();
        float dy = y.size();
        float dz = z.size();
        return 2.0f * (dx * dy + dy * dz + dz * dx);
    }

private:
    __host__ __device__ void pad_to_minimums() {
        float delta = 0.0001f;
        if (x.size() < delta) x = x.expand(delta);
        if (y.size() < delta) y = y.expand(delta);
        if (z.size() < delta) z = z.expand(delta);
    }
};

}

#endif
