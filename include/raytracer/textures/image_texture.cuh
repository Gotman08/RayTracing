#ifndef RAYTRACER_TEXTURES_IMAGE_TEXTURE_CUH
#define RAYTRACER_TEXTURES_IMAGE_TEXTURE_CUH

#include "raytracer/textures/texture.cuh"

namespace rt {

class ImageTexture : public Texture {
public:
    unsigned char* data;
    int width;
    int height;
    int channels;
    int bytes_per_scanline;

    __host__ __device__ ImageTexture()
        : Texture(TextureType::IMAGE), data(nullptr), width(0), height(0), channels(0) {}

    __host__ __device__ ImageTexture(unsigned char* d, int w, int h, int c)
        : Texture(TextureType::IMAGE), data(d), width(w), height(h), channels(c) {
        bytes_per_scanline = width * channels;
    }

    __host__ __device__ Color value(float u, float v, const Point3& p) const {
        if (data == nullptr)
            return Color(0, 1, 1); // Cyan for debugging

        // Clamp UV coordinates
        u = u < 0 ? 0 : (u > 1 ? 1 : u);
        v = 1.0f - v; // Flip V (image is stored top to bottom)
        v = v < 0 ? 0 : (v > 1 ? 1 : v);

        int i = static_cast<int>(u * width);
        int j = static_cast<int>(v * height);

        if (i >= width) i = width - 1;
        if (j >= height) j = height - 1;

        const float color_scale = 1.0f / 255.0f;
        unsigned char* pixel = data + j * bytes_per_scanline + i * channels;

        return Color(
            color_scale * pixel[0],
            color_scale * pixel[1],
            color_scale * pixel[2]
        );
    }
};

} // namespace rt

#endif // RAYTRACER_TEXTURES_IMAGE_TEXTURE_CUH
