#ifndef RAYTRACER_CORE_RANDOM_CUH
#define RAYTRACER_CORE_RANDOM_CUH

#include <random>
#include <cstdint>
#include <cmath>

namespace rt {

// Forward declarations
class Vec3;

class CPURandom {
public:
    CPURandom(uint64_t seed = 0) : gen(seed), dist(0.0f, 1.0f) {}

    float operator()() {
        return dist(gen);
    }

    float random_float() {
        return dist(gen);
    }

    float random_float(float min, float max) {
        return min + (max - min) * dist(gen);
    }

private:
    std::mt19937 gen;
    std::uniform_real_distribution<float> dist;
};

}

// Include Vec3 after namespace closure to avoid issues
#include "raytracer/core/vec3.cuh"

namespace rt {

constexpr float RANDOM_TWO_PI = 2.0f * 3.14159265358979323846f;

inline Vec3 random_vec3(CPURandom& rng) {
    return Vec3(rng(), rng(), rng());
}

inline Vec3 random_vec3(float min, float max, CPURandom& rng) {
    float range = max - min;
    return Vec3(
        min + range * rng(),
        min + range * rng(),
        min + range * rng()
    );
}

inline Vec3 random_unit_vector(CPURandom& rng) {
    float z = 2.0f * rng() - 1.0f;
    float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    float phi = RANDOM_TWO_PI * rng();
    return Vec3(r * cosf(phi), r * sinf(phi), z);
}

inline Vec3 random_in_unit_sphere(CPURandom& rng) {
    Vec3 dir = random_unit_vector(rng);
    float r = cbrtf(rng());
    return dir * r;
}

inline Vec3 random_on_hemisphere(const Vec3& normal, CPURandom& rng) {
    Vec3 on_unit_sphere = random_unit_vector(rng);
    if (dot(on_unit_sphere, normal) > 0.0f)
        return on_unit_sphere;
    else
        return -on_unit_sphere;
}

inline Vec3 random_in_unit_disk(CPURandom& rng) {
    float r = sqrtf(rng());
    float theta = RANDOM_TWO_PI * rng();
    return Vec3(r * cosf(theta), r * sinf(theta), 0.0f);
}

inline Vec3 random_cosine_direction(CPURandom& rng) {
    float r1 = rng();
    float r2 = rng();

    float phi = RANDOM_TWO_PI * r1;
    float x = cosf(phi) * sqrtf(r2);
    float y = sinf(phi) * sqrtf(r2);
    float z = sqrtf(1.0f - r2);

    return Vec3(x, y, z);
}

}

#endif
