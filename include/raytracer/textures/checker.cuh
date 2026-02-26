#ifndef RAYTRACER_TEXTURES_CHECKER_CUH
#define RAYTRACER_TEXTURES_CHECKER_CUH

#include "raytracer/textures/texture.cuh"

namespace rt {

class CheckerTexture : public Texture {
public:
    float inv_scale;
    Color even_color;
    Color odd_color;

    __host__ __device__ CheckerTexture()
        : Texture(TextureType::CHECKER), inv_scale(1.0f),
          even_color(0, 0, 0), odd_color(1, 1, 1) {}

    __host__ __device__ CheckerTexture(float scale, const Color& c1, const Color& c2)
        : Texture(TextureType::CHECKER), inv_scale(1.0f / scale),
          even_color(c1), odd_color(c2) {}

    __host__ __device__ Color value(float u, float v, const Point3& p) const {
        int xi = static_cast<int>(floorf(inv_scale * p.x));
        int yi = static_cast<int>(floorf(inv_scale * p.y));
        int zi = static_cast<int>(floorf(inv_scale * p.z));

        bool is_even = (xi + yi + zi) % 2 == 0;
        return is_even ? even_color : odd_color;
    }
};

} // namespace rt

#endif // RAYTRACER_TEXTURES_CHECKER_CUH
