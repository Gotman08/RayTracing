#ifndef RAYTRACER_TEXTURES_NOISE_CUH
#define RAYTRACER_TEXTURES_NOISE_CUH

#include "raytracer/textures/texture.cuh"

namespace rt {

// GPU-friendly Perlin noise with static arrays
class PerlinGPU {
public:
    static const int POINT_COUNT = 256;
    Vec3 ranvec[POINT_COUNT];
    int perm_x[POINT_COUNT];
    int perm_y[POINT_COUNT];
    int perm_z[POINT_COUNT];
    bool initialized;

    __host__ __device__ PerlinGPU() : initialized(false) {}

    // Host-only initialization
    __host__ void initialize(unsigned int seed = 42) {
        srand(seed);

        // Generate random vectors
        for (int i = 0; i < POINT_COUNT; i++) {
            ranvec[i] = Vec3(
                (float)rand() / RAND_MAX * 2 - 1,
                (float)rand() / RAND_MAX * 2 - 1,
                (float)rand() / RAND_MAX * 2 - 1
            ).normalized();
        }

        // Generate permutation tables
        for (int i = 0; i < POINT_COUNT; i++) {
            perm_x[i] = i;
            perm_y[i] = i;
            perm_z[i] = i;
        }

        // Shuffle each permutation table
        permute(perm_x, POINT_COUNT);
        permute(perm_y, POINT_COUNT);
        permute(perm_z, POINT_COUNT);

        initialized = true;
    }

    __host__ __device__ float noise(const Point3& p) const {
        float u = p.x - floorf(p.x);
        float v = p.y - floorf(p.y);
        float w = p.z - floorf(p.z);

        int i = static_cast<int>(floorf(p.x));
        int j = static_cast<int>(floorf(p.y));
        int k = static_cast<int>(floorf(p.z));
        Vec3 c[2][2][2];

        for (int di = 0; di < 2; di++) {
            for (int dj = 0; dj < 2; dj++) {
                for (int dk = 0; dk < 2; dk++) {
                    c[di][dj][dk] = ranvec[
                        perm_x[(i + di) & 255] ^
                        perm_y[(j + dj) & 255] ^
                        perm_z[(k + dk) & 255]
                    ];
                }
            }
        }

        return perlin_interp(c, u, v, w);
    }

    __host__ __device__ float turb(const Point3& p, int depth = 7) const {
        float accum = 0.0f;
        Point3 temp_p = p;
        float weight = 1.0f;

        for (int i = 0; i < depth; i++) {
            accum += weight * noise(temp_p);
            weight *= 0.5f;
            temp_p = temp_p * 2.0f;
        }

        return fabsf(accum);
    }

private:
    __host__ static void permute(int* p, int n) {
        for (int i = n - 1; i > 0; i--) {
            int target = rand() % (i + 1);
            int tmp = p[i];
            p[i] = p[target];
            p[target] = tmp;
        }
    }

    __host__ __device__ static float perlin_interp(const Vec3 c[2][2][2], float u, float v, float w) {
        float uu = u * u * (3 - 2 * u);
        float vv = v * v * (3 - 2 * v);
        float ww = w * w * (3 - 2 * w);
        float accum = 0.0f;

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    Vec3 weight_v(u - i, v - j, w - k);
                    accum += (i * uu + (1 - i) * (1 - uu))
                           * (j * vv + (1 - j) * (1 - vv))
                           * (k * ww + (1 - k) * (1 - ww))
                           * dot(c[i][j][k], weight_v);
                }
            }
        }

        return accum;
    }
};

// GPU-friendly NoiseTexture that embeds the Perlin data
class NoiseTexture : public Texture {
public:
    PerlinGPU perlin;
    float scale;
    Color base_color;

    __host__ __device__ NoiseTexture()
        : Texture(TextureType::NOISE), scale(1.0f), base_color(1, 1, 1) {}

    __host__ NoiseTexture(float s, const Color& c = Color(1, 1, 1), unsigned int seed = 42)
        : Texture(TextureType::NOISE), scale(s), base_color(c) {
        perlin.initialize(seed);
    }

    __host__ __device__ Color value(float u, float v, const Point3& p) const {
        if (!perlin.initialized) return base_color;
        return base_color * 0.5f * (1.0f + sinf(scale * p.z + 10.0f * perlin.turb(p)));
    }
};

} // namespace rt

#endif // RAYTRACER_TEXTURES_NOISE_CUH
