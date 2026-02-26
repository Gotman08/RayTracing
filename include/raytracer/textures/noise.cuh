#ifndef RAYTRACER_TEXTURES_NOISE_CUH
#define RAYTRACER_TEXTURES_NOISE_CUH

#include "raytracer/textures/texture.cuh"

namespace rt {

// Perlin noise implementation
class Perlin {
public:
    static const int point_count = 256;
    Vec3* ranvec;
    int* perm_x;
    int* perm_y;
    int* perm_z;

    __host__ Perlin() {
        ranvec = new Vec3[point_count];
        for (int i = 0; i < point_count; i++) {
            ranvec[i] = Vec3(
                (float)rand() / RAND_MAX * 2 - 1,
                (float)rand() / RAND_MAX * 2 - 1,
                (float)rand() / RAND_MAX * 2 - 1
            ).normalized();
        }

        perm_x = perlin_generate_perm();
        perm_y = perlin_generate_perm();
        perm_z = perlin_generate_perm();
    }

    __host__ ~Perlin() {
        delete[] ranvec;
        delete[] perm_x;
        delete[] perm_y;
        delete[] perm_z;
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
    __host__ static int* perlin_generate_perm() {
        int* p = new int[point_count];
        for (int i = 0; i < point_count; i++) {
            p[i] = i;
        }
        permute(p, point_count);
        return p;
    }

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

class NoiseTexture : public Texture {
public:
    Perlin* noise;
    float scale;
    Color base_color;

    __host__ __device__ NoiseTexture()
        : Texture(TextureType::NOISE), noise(nullptr), scale(1.0f), base_color(1, 1, 1) {}

    __host__ __device__ NoiseTexture(Perlin* n, float s, const Color& c = Color(1, 1, 1))
        : Texture(TextureType::NOISE), noise(n), scale(s), base_color(c) {}

    __host__ __device__ Color value(float u, float v, const Point3& p) const {
        if (!noise) return base_color;
        return base_color * 0.5f * (1.0f + sinf(scale * p.z + 10.0f * noise->turb(p)));
    }
};

} // namespace rt

#endif // RAYTRACER_TEXTURES_NOISE_CUH
