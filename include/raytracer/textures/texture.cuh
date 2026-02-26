#ifndef RAYTRACER_TEXTURES_TEXTURE_CUH
#define RAYTRACER_TEXTURES_TEXTURE_CUH

#include "raytracer/core/vec3.cuh"

namespace rt {

enum class TextureType {
    SOLID_COLOR,
    CHECKER,
    NOISE,
    IMAGE
};

class Texture {
public:
    TextureType type;

    __host__ __device__ Texture() : type(TextureType::SOLID_COLOR) {}
    __host__ __device__ Texture(TextureType t) : type(t) {}
};

} // namespace rt

#endif // RAYTRACER_TEXTURES_TEXTURE_CUH
