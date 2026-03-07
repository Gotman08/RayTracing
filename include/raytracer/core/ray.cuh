#ifndef RAYTRACER_CORE_RAY_CUH
#define RAYTRACER_CORE_RAY_CUH

#include "raytracer/core/vec3.cuh"

namespace rt {

class Ray {
public:
    Point3 orig;
    Vec3 dir;
    float tm;

    __host__ __device__ Ray() : orig(), dir(), tm(0.0f) {}

    __host__ __device__ Ray(const Point3& origin, const Vec3& direction, float time = 0.0f)
        : orig(origin), dir(direction), tm(time) {}

    __host__ __device__ const Point3& origin() const { return orig; }
    __host__ __device__ const Vec3& direction() const { return dir; }
    __host__ __device__ float time() const { return tm; }

    __host__ __device__ Point3 at(float t) const {
        return orig + t * dir;
    }
};

}

#endif
