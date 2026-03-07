#ifndef RAYTRACER_GEOMETRY_GEOMETRY_UTILS_CUH
#define RAYTRACER_GEOMETRY_GEOMETRY_UTILS_CUH

#include "raytracer/core/vec3.cuh"
#include "raytracer/core/cuda_utils.cuh"

namespace rt {

__host__ __device__ inline void get_sphere_uv(const Vec3& p, float& u, float& v) {
    float theta = acosf(-p.y);
    float phi = atan2f(-p.z, p.x) + PI;
    u = phi / (2.0f * PI);
    v = theta / PI;
}

}

#endif
