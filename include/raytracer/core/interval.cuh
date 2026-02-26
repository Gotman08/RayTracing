#ifndef RAYTRACER_CORE_INTERVAL_CUH
#define RAYTRACER_CORE_INTERVAL_CUH

#include <cuda_runtime.h>
#include <cfloat>

namespace rt {

class Interval {
public:
    float min, max;

    __host__ __device__ Interval() : min(+FLT_MAX), max(-FLT_MAX) {} // Empty interval

    __host__ __device__ Interval(float _min, float _max) : min(_min), max(_max) {}

    __host__ __device__ Interval(const Interval& a, const Interval& b) {
        min = a.min <= b.min ? a.min : b.min;
        max = a.max >= b.max ? a.max : b.max;
    }

    __host__ __device__ float size() const {
        return max - min;
    }

    __host__ __device__ bool contains(float x) const {
        return min <= x && x <= max;
    }

    __host__ __device__ bool surrounds(float x) const {
        return min < x && x < max;
    }

    __host__ __device__ float clamp(float x) const {
        if (x < min) return min;
        if (x > max) return max;
        return x;
    }

    __host__ __device__ Interval expand(float delta) const {
        float padding = delta / 2.0f;
        return Interval(min - padding, max + padding);
    }

    static const Interval empty;
    static const Interval universe;
};

// Static interval definitions
__device__ __constant__ inline Interval Interval_empty(+FLT_MAX, -FLT_MAX);
__device__ __constant__ inline Interval Interval_universe(-FLT_MAX, +FLT_MAX);

} // namespace rt

#endif // RAYTRACER_CORE_INTERVAL_CUH
