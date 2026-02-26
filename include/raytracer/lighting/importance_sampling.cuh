#ifndef RAYTRACER_LIGHTING_IMPORTANCE_SAMPLING_CUH
#define RAYTRACER_LIGHTING_IMPORTANCE_SAMPLING_CUH

#include "raytracer/lighting/pdf.cuh"

namespace rt {

// Light sampling utilities
struct LightSample {
    Point3 position;
    Vec3 direction;
    Color emission;
    float pdf;
    float distance;
};

// Sample a point light
__device__ inline LightSample sample_point_light(
    const Point3& light_pos,
    const Color& light_color,
    const Point3& hit_point
) {
    LightSample sample;
    sample.direction = light_pos - hit_point;
    sample.distance = sample.direction.length();
    sample.direction = sample.direction / sample.distance;
    sample.position = light_pos;
    sample.emission = light_color;
    sample.pdf = 1.0f;  // Delta distribution
    return sample;
}

// Sample an area light (quad)
__device__ inline LightSample sample_quad_light(
    const Point3& Q,
    const Vec3& u,
    const Vec3& v,
    const Color& emission,
    const Point3& hit_point,
    curandState* rand_state
) {
    LightSample sample;

    // Random point on quad
    float r1 = curand_uniform(rand_state);
    float r2 = curand_uniform(rand_state);
    sample.position = Q + r1 * u + r2 * v;

    sample.direction = sample.position - hit_point;
    sample.distance = sample.direction.length();
    sample.direction = sample.direction / sample.distance;

    // Compute PDF (1 / area * geometry factor)
    Vec3 normal = unit_vector(cross(u, v));
    float area = cross(u, v).length();
    float cos_theta = fabsf(dot(normal, -sample.direction));

    if (cos_theta < 0.0001f) {
        sample.pdf = 0;
        sample.emission = Color(0, 0, 0);
    } else {
        sample.pdf = (sample.distance * sample.distance) / (cos_theta * area);
        sample.emission = emission;
    }

    return sample;
}

// Multiple Importance Sampling weight (balance heuristic)
__host__ __device__ inline float power_heuristic(int nf, float fPdf, int ng, float gPdf) {
    float f = nf * fPdf;
    float g = ng * gPdf;
    return (f * f) / (f * f + g * g);
}

__host__ __device__ inline float balance_heuristic(int nf, float fPdf, int ng, float gPdf) {
    float f = nf * fPdf;
    float g = ng * gPdf;
    return f / (f + g);
}

} // namespace rt

#endif // RAYTRACER_LIGHTING_IMPORTANCE_SAMPLING_CUH
