#ifndef RAYTRACER_LIGHTING_PDF_CUH
#define RAYTRACER_LIGHTING_PDF_CUH

#include "raytracer/core/vec3.cuh"
#include "raytracer/core/cuda_utils.cuh"

namespace rt {

// Orthonormal Basis for coordinate transforms
class ONB {
public:
    Vec3 u, v, w;

    __host__ __device__ ONB() {}

    __host__ __device__ void build_from_w(const Vec3& n) {
        w = unit_vector(n);
        Vec3 a = (fabsf(w.x) > 0.9f) ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
        v = unit_vector(cross(w, a));
        u = cross(w, v);
    }

    __host__ __device__ Vec3 local(float a, float b, float c) const {
        return a * u + b * v + c * w;
    }

    __host__ __device__ Vec3 local(const Vec3& a) const {
        return a.x * u + a.y * v + a.z * w;
    }
};

// Cosine-weighted PDF for diffuse surfaces
class CosinePDF {
public:
    ONB uvw;

    __host__ __device__ CosinePDF() {}

    __host__ __device__ CosinePDF(const Vec3& w) {
        uvw.build_from_w(w);
    }

    __host__ __device__ float value(const Vec3& direction) const {
        float cosine_theta = dot(unit_vector(direction), uvw.w);
        return fmaxf(0.0f, cosine_theta / PI);
    }

    __device__ Vec3 generate(curandState* rand_state) const {
        return uvw.local(random_cosine_direction(rand_state));
    }
};

// Sphere PDF (for area lights)
class SpherePDF {
public:
    Point3 center;
    float radius;

    __host__ __device__ SpherePDF() : radius(0) {}

    __host__ __device__ SpherePDF(const Point3& c, float r) : center(c), radius(r) {}

    __host__ __device__ float value(const Vec3& direction, const Point3& origin) const {
        // ... simplified implementation
        return 1.0f / (4.0f * PI);
    }

    __device__ Vec3 generate(const Point3& origin, curandState* rand_state) const {
        Vec3 direction = center - origin;
        float dist_sq = direction.length_squared();
        ONB uvw;
        uvw.build_from_w(direction);
        return uvw.local(random_to_sphere(radius, dist_sq, rand_state));
    }

private:
    __device__ static Vec3 random_to_sphere(float radius, float dist_sq, curandState* rand_state) {
        float r1 = curand_uniform(rand_state);
        float r2 = curand_uniform(rand_state);
        float z = 1.0f + r2 * (sqrtf(1.0f - radius * radius / dist_sq) - 1.0f);

        float phi = 2.0f * PI * r1;
        float x = cosf(phi) * sqrtf(1.0f - z * z);
        float y = sinf(phi) * sqrtf(1.0f - z * z);

        return Vec3(x, y, z);
    }
};

// Mixture PDF for combining different sampling strategies
class MixturePDF {
public:
    float weight;  // Weight for first PDF (1-weight for second)

    __host__ __device__ MixturePDF() : weight(0.5f) {}

    __host__ __device__ MixturePDF(float w) : weight(w) {}
};

} // namespace rt

#endif // RAYTRACER_LIGHTING_PDF_CUH
