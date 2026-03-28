#ifndef RAYTRACER_CORE_INTERVAL_CUH
#define RAYTRACER_CORE_INTERVAL_CUH

/** @file interval.cuh
 *  @brief Classe Interval - plage [min, max] flottante */

#include <cfloat>
#include "raytracer/core/cuda_compat.cuh"

namespace rt {

/** @brief Intervalle ferme [min, max] - vide par defaut */
class Interval {
public:
    float min;
    float max;

    /** @brief Ctor par defaut -> intervalle vide */
    __host__ __device__ Interval() : min(+FLT_MAX), max(-FLT_MAX) {}

    /** @brief Ctor (min, max) */
    __host__ __device__ Interval(float _min, float _max) : min(_min), max(_max) {}

    /** @brief Union de deux intervalles */
    __host__ __device__ Interval(const Interval& a, const Interval& b) {
        min = a.min <= b.min ? a.min : b.min;
        max = a.max >= b.max ? a.max : b.max;
    }

    /** @brief Taille (max - min) */
    __host__ __device__ float size() const {
        return max - min;
    }

    /** @brief Test min <= x <= max */
    __host__ __device__ bool contains(float x) const {
        return min <= x && x <= max;
    }

    /** @brief Test strict min < x < max */
    __host__ __device__ bool surrounds(float x) const {
        return min < x && x < max;
    }

    /** @brief Clamp x dans [min, max] */
    __host__ __device__ float clamp(float x) const {
        if (x < min) return min;
        if (x > max) return max;
        return x;
    }

    /** @brief Elargit de delta/2 de chaque cote */
    __host__ __device__ Interval expand(float delta) const {
        float padding = delta / 2.0f;
        return Interval(min - padding, max + padding);
    }

};


}

#endif
