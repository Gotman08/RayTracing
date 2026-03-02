#ifndef RAYTRACER_ENVIRONMENT_HDRI_CUH
#define RAYTRACER_ENVIRONMENT_HDRI_CUH

#include "raytracer/core/vec3.cuh"
#include "raytracer/core/cuda_utils.cuh"

namespace rt {

// HDR Environment Map
class HDRI {
public:
    float* data;          // RGB float data
    int width;
    int height;
    float intensity;
    float rotation;       // Y-axis rotation in radians

    __host__ __device__ HDRI()
        : data(nullptr), width(0), height(0), intensity(1.0f), rotation(0.0f) {}

    __host__ __device__ HDRI(float* d, int w, int h, float i = 1.0f)
        : data(d), width(w), height(h), intensity(i), rotation(0.0f) {}

    __host__ __device__ Color sample(const Vec3& direction) const {
        if (data == nullptr)
            return Color(0, 0, 0);

        Vec3 dir = unit_vector(direction);

        // Apply rotation around Y axis
        if (rotation != 0.0f) {
            float cos_r = cosf(rotation);
            float sin_r = sinf(rotation);
            float new_x = cos_r * dir.x + sin_r * dir.z;
            float new_z = -sin_r * dir.x + cos_r * dir.z;
            dir.x = new_x;
            dir.z = new_z;
        }

        // Convert direction to spherical coordinates
        float phi = atan2f(dir.z, dir.x);  // [-PI, PI]
        float theta = acosf(dir.y);         // [0, PI]

        // Convert to UV coordinates
        float u = (phi + PI) / (2.0f * PI);
        float v = theta / PI;

        // Sample texture
        int i = static_cast<int>(u * width);
        int j = static_cast<int>(v * height);

        if (i >= width) i = width - 1;
        if (j >= height) j = height - 1;
        if (i < 0) i = 0;
        if (j < 0) j = 0;

        int idx = (j * width + i) * 3;

        return intensity * Color(data[idx], data[idx + 1], data[idx + 2]);
    }

    // Bilinear sampling for smoother results
    __host__ __device__ Color sample_bilinear(const Vec3& direction) const {
        if (data == nullptr)
            return Color(0, 0, 0);

        Vec3 dir = unit_vector(direction);

        // Apply rotation
        if (rotation != 0.0f) {
            float cos_r = cosf(rotation);
            float sin_r = sinf(rotation);
            float new_x = cos_r * dir.x + sin_r * dir.z;
            float new_z = -sin_r * dir.x + cos_r * dir.z;
            dir.x = new_x;
            dir.z = new_z;
        }

        float phi = atan2f(dir.z, dir.x);
        float theta = acosf(dir.y);

        float u = (phi + PI) / (2.0f * PI);
        float v = theta / PI;

        float fx = u * width - 0.5f;
        float fy = v * height - 0.5f;

        int x0 = static_cast<int>(floorf(fx));
        int y0 = static_cast<int>(floorf(fy));
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        float dx = fx - x0;
        float dy = fy - y0;

        // Wrap coordinates
        x0 = (x0 % width + width) % width;
        x1 = (x1 % width + width) % width;
        y0 = (y0 < 0) ? 0 : (y0 >= height ? height - 1 : y0);
        y1 = (y1 < 0) ? 0 : (y1 >= height ? height - 1 : y1);

        Color c00 = get_pixel(x0, y0);
        Color c10 = get_pixel(x1, y0);
        Color c01 = get_pixel(x0, y1);
        Color c11 = get_pixel(x1, y1);

        Color c0 = (1 - dx) * c00 + dx * c10;
        Color c1 = (1 - dx) * c01 + dx * c11;

        return intensity * ((1 - dy) * c0 + dy * c1);
    }

private:
    __host__ __device__ Color get_pixel(int x, int y) const {
        int idx = (y * width + x) * 3;
        return Color(data[idx], data[idx + 1], data[idx + 2]);
    }
};

} // namespace rt

#endif // RAYTRACER_ENVIRONMENT_HDRI_CUH
