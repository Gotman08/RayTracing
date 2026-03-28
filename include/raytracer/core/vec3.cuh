#ifndef RAYTRACER_CORE_VEC3_CUH
#define RAYTRACER_CORE_VEC3_CUH

/** @file vec3.cuh
 *  @brief Classe Vec3 - vecteur 3D (positions, directions, couleurs) */

#include <cmath>
#include "raytracer/core/cuda_compat.cuh"

#ifdef __CUDACC__
#include <curand_kernel.h>
#endif

namespace rt {

constexpr float VEC3_PI = 3.14159265358979323846f;
constexpr float VEC3_TWO_PI = 2.0f * VEC3_PI;

/** @brief Vecteur 3D flottant - base du raytracer
 *  Aligne a 16 octets pour coalescence GPU (transactions 128-bit) */
class __align__(16) Vec3 {
public:
    float x, y, z;

    /** @brief Ctor par defaut -> (0,0,0) */
    __host__ __device__ Vec3() : x(0.0f), y(0.0f), z(0.0f) {}

    /** @brief Ctor (x, y, z) */
    __host__ __device__ Vec3(float e0, float e1, float e2) : x(e0), y(e1), z(e2) {}

    /** @brief Ctor broadcast - meme valeur pour x,y,z */
    __host__ __device__ explicit Vec3(float v) : x(v), y(v), z(v) {}

    /** @brief Negation unaire */
    __host__ __device__ Vec3 operator-() const { return Vec3(-x, -y, -z); }

    /** @brief Acces par indice (0=x, 1=y, 2=z) */
    __host__ __device__ float operator[](int i) const {
        if (i == 0) return x;
        if (i == 1) return y;
        return z;
    }

    /** @brief Addition composante par composante */
    __host__ __device__ Vec3 operator+(const Vec3& v) const {
        return Vec3(x + v.x, y + v.y, z + v.z);
    }

    /** @brief Soustraction composante par composante */
    __host__ __device__ Vec3 operator-(const Vec3& v) const {
        return Vec3(x - v.x, y - v.y, z - v.z);
    }

    /** @brief Produit de Hadamard (modulation couleurs) */
    __host__ __device__ Vec3 operator*(const Vec3& v) const {
        return Vec3(x * v.x, y * v.y, z * v.z);
    }

    /** @brief Mul par scalaire */
    __host__ __device__ Vec3 operator*(float t) const {
        return Vec3(x * t, y * t, z * t);
    }

    /** @brief Div par scalaire */
    __host__ __device__ Vec3 operator/(float t) const {
        return Vec3(x / t, y / t, z / t);
    }

    /** @brief += en place */
    __host__ __device__ Vec3& operator+=(const Vec3& v) {
        x += v.x; y += v.y; z += v.z;
        return *this;
    }

    /** @brief *= en place */
    __host__ __device__ Vec3& operator*=(float t) {
        x *= t; y *= t; z *= t;
        return *this;
    }

    /** @brief /= en place */
    __host__ __device__ Vec3& operator/=(float t) {
        return *this *= 1.0f / t;
    }

    /** @brief Norme euclidienne */
    __host__ __device__ float length() const {
        return sqrtf(x*x + y*y + z*z);
    }

    /** @brief Norme au carre (evite le sqrt) */
    __host__ __device__ float length_squared() const {
        return x*x + y*y + z*z;
    }

    /** @brief Retourne le vecteur unitaire (safe si norme nulle) */
    __host__ __device__ Vec3 normalized() const {
        float len = length();
        return (len > 0.0f) ? (*this / len) : Vec3();
    }

    /** @brief Vrai si quasi-nul (seuil 1e-8) */
    __host__ __device__ bool near_zero() const {
        const float s = 1e-8f;
        return (fabsf(x) < s) && (fabsf(y) < s) && (fabsf(z) < s);
    }
};

using Point3 = Vec3; ///< Position 3D
using Color = Vec3;  ///< Couleur RGB

/** @brief scalaire * vecteur (commutativite) */
__host__ __device__ inline Vec3 operator*(float t, const Vec3& v) {
    return v * t;
}

/** @brief Produit scalaire u.v */
__host__ __device__ inline float dot(const Vec3& u, const Vec3& v) {
    return u.x*v.x + u.y*v.y + u.z*v.z;
}

/** @brief Produit vectoriel u x v */
__host__ __device__ inline Vec3 cross(const Vec3& u, const Vec3& v) {
    return Vec3(
        u.y*v.z - u.z*v.y,
        u.z*v.x - u.x*v.z,
        u.x*v.y - u.y*v.x
    );
}

/** @brief Reflexion de v par rapport a n */
__host__ __device__ inline Vec3 reflect(const Vec3& v, const Vec3& n) {
    return v - 2.0f * dot(v, n) * n;
}

/** @brief Refraction Snell-Descartes (uv unitaire, n unitaire) */
__host__ __device__ inline Vec3 refract(const Vec3& uv, const Vec3& n, float etai_over_etat) {
    float cos_theta = fminf(dot(-uv, n), 1.0f);
    Vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    Vec3 r_out_parallel = -sqrtf(fabsf(1.0f - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

/** @brief Wrapper normalize */
__host__ __device__ inline Vec3 unit_vector(const Vec3& v) {
    return v.normalized();
}

#ifdef __CUDACC__
/** @brief Vec3 aleatoire GPU dans [0,1] */
__device__ inline Vec3 random_vec3(curandState* rand_state) {
    return Vec3(
        curand_uniform(rand_state),
        curand_uniform(rand_state),
        curand_uniform(rand_state)
    );
}

/** @brief Vec3 aleatoire GPU dans [min,max] */
__device__ inline Vec3 random_vec3(float min, float max, curandState* rand_state) {
    float range = max - min;
    return Vec3(
        min + range * curand_uniform(rand_state),
        min + range * curand_uniform(rand_state),
        min + range * curand_uniform(rand_state)
    );
}

/** @brief Dir unitaire aleatoire uniforme sur la sphere */
__device__ inline Vec3 random_unit_vector(curandState* rand_state) {
    float z = 2.0f * curand_uniform(rand_state) - 1.0f;
    float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    float phi = VEC3_TWO_PI * curand_uniform(rand_state);
    return Vec3(r * cosf(phi), r * sinf(phi), z);
}

/** @brief Point aleatoire dans la sphere unite */
__device__ inline Vec3 random_in_unit_sphere(curandState* rand_state) {
    Vec3 dir = random_unit_vector(rand_state);
    float r = cbrtf(curand_uniform(rand_state));
    return dir * r;
}

/** @brief Dir aleatoire dans l'hemisphere de la normale */
__device__ inline Vec3 random_on_hemisphere(const Vec3& normal, curandState* rand_state) {
    Vec3 on_unit_sphere = random_unit_vector(rand_state);
    if (dot(on_unit_sphere, normal) > 0.0f)
        return on_unit_sphere;
    else
        return -on_unit_sphere;
}

/** @brief Point aleatoire dans le disque unite (defocus blur) */
__device__ inline Vec3 random_in_unit_disk(curandState* rand_state) {
    float r = sqrtf(curand_uniform(rand_state));
    float theta = VEC3_TWO_PI * curand_uniform(rand_state);
    return Vec3(r * cosf(theta), r * sinf(theta), 0.0f);
}

/** @brief Dir cosine-weighted (lambertien optimal) */
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
