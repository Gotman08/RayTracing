#ifndef RAYTRACER_CORE_VEC3_CUH
#define RAYTRACER_CORE_VEC3_CUH

#include <cmath>
#include "raytracer/core/cuda_compat.cuh"

#ifdef __CUDACC__
#include <curand_kernel.h>
#endif

namespace rt {

constexpr float VEC3_PI = 3.14159265358979323846f;
constexpr float VEC3_TWO_PI = 2.0f * VEC3_PI;

class Vec3 {
public:
    float x, y, z;

    __host__ __device__ Vec3() : x(0.0f), y(0.0f), z(0.0f) {}
    __host__ __device__ Vec3(float e0, float e1, float e2) : x(e0), y(e1), z(e2) {}
    __host__ __device__ explicit Vec3(float v) : x(v), y(v), z(v) {}

    __host__ __device__ Vec3 operator-() const { return Vec3(-x, -y, -z); }

    __host__ __device__ float operator[](int i) const {
        if (i == 0) return x;
        if (i == 1) return y;
        return z;
    }

    __host__ __device__ Vec3 operator+(const Vec3& v) const {
        return Vec3(x + v.x, y + v.y, z + v.z);
    }

    __host__ __device__ Vec3 operator-(const Vec3& v) const {
        return Vec3(x - v.x, y - v.y, z - v.z);
    }

    __host__ __device__ Vec3 operator*(const Vec3& v) const {
        return Vec3(x * v.x, y * v.y, z * v.z);
    }

    __host__ __device__ Vec3 operator*(float t) const {
        return Vec3(x * t, y * t, z * t);
    }

    __host__ __device__ Vec3 operator/(float t) const {
        return Vec3(x / t, y / t, z / t);
    }

    __host__ __device__ Vec3& operator+=(const Vec3& v) {
        x += v.x; y += v.y; z += v.z;
        return *this;
    }

    __host__ __device__ Vec3& operator*=(float t) {
        x *= t; y *= t; z *= t;
        return *this;
    }

    __host__ __device__ Vec3& operator/=(float t) {
        return *this *= 1.0f / t;
    }

    __host__ __device__ float length() const {
        return sqrtf(x*x + y*y + z*z);
    }

    __host__ __device__ float length_squared() const {
        return x*x + y*y + z*z;
    }

    __host__ __device__ Vec3 normalized() const {
        float len = length();
        return (len > 0.0f) ? (*this / len) : Vec3();
    }

    __host__ __device__ bool near_zero() const {
        const float s = 1e-8f;
        return (fabsf(x) < s) && (fabsf(y) < s) && (fabsf(z) < s);
    }
};

using Point3 = Vec3;
using Color = Vec3;

__host__ __device__ inline Vec3 operator*(float t, const Vec3& v) {
    return v * t;
}

__host__ __device__ inline float dot(const Vec3& u, const Vec3& v) {
    return u.x*v.x + u.y*v.y + u.z*v.z;
}

__host__ __device__ inline Vec3 cross(const Vec3& u, const Vec3& v) {
    return Vec3(
        u.y*v.z - u.z*v.y,
        u.z*v.x - u.x*v.z,
        u.x*v.y - u.y*v.x
    );
}

__host__ __device__ inline Vec3 reflect(const Vec3& v, const Vec3& n) {
    return v - 2.0f * dot(v, n) * n;
}

__host__ __device__ inline Vec3 refract(const Vec3& uv, const Vec3& n, float etai_over_etat) {
    float cos_theta = fminf(dot(-uv, n), 1.0f);
    Vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    Vec3 r_out_parallel = -sqrtf(fabsf(1.0f - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

__host__ __device__ inline Vec3 unit_vector(const Vec3& v) {
    return v.normalized();
}

#ifdef __CUDACC__
__device__ inline Vec3 random_vec3(curandState* rand_state) {
    return Vec3(
        curand_uniform(rand_state),
        curand_uniform(rand_state),
        curand_uniform(rand_state)
    );
}

__device__ inline Vec3 random_vec3(float min, float max, curandState* rand_state) {
    float range = max - min;
    return Vec3(
        min + range * curand_uniform(rand_state),
        min + range * curand_uniform(rand_state),
        min + range * curand_uniform(rand_state)
    );
}

__device__ inline Vec3 random_unit_vector(curandState* rand_state) {
    float z = 2.0f * curand_uniform(rand_state) - 1.0f;
    float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    float phi = VEC3_TWO_PI * curand_uniform(rand_state);
    return Vec3(r * cosf(phi), r * sinf(phi), z);
}

__device__ inline Vec3 random_in_unit_sphere(curandState* rand_state) {
    Vec3 dir = random_unit_vector(rand_state);
    float r = cbrtf(curand_uniform(rand_state));
    return dir * r;
}

__device__ inline Vec3 random_on_hemisphere(const Vec3& normal, curandState* rand_state) {
    Vec3 on_unit_sphere = random_unit_vector(rand_state);
    if (dot(on_unit_sphere, normal) > 0.0f)
        return on_unit_sphere;
    else
        return -on_unit_sphere;
}

__device__ inline Vec3 random_in_unit_disk(curandState* rand_state) {
    float r = sqrtf(curand_uniform(rand_state));
    float theta = VEC3_TWO_PI * curand_uniform(rand_state);
    return Vec3(r * cosf(theta), r * sinf(theta), 0.0f);
}

__device__ inline Vec3 random_cosine_direction(curandState* rand_state) {
    float r1 = curand_uniform(rand_state);
    float r2 = curand_uniform(rand_state);

    float phi = 2.0f * 3.14159265358979323846f * r1;
    float x = cosf(phi) * sqrtf(r2);
    float y = sinf(phi) * sqrtf(r2);
    float z = sqrtf(1.0f - r2);

    return Vec3(x, y, z);
}
#endif

}

#endif
