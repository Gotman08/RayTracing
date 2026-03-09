#ifndef RAYTRACER_INTERACTIVE_GL_INTEROP_CUH
#define RAYTRACER_INTERACTIVE_GL_INTEROP_CUH

#ifdef ENABLE_INTERACTIVE

#include <GL/glew.h>
#include <GL/gl.h>
#include <cuda_gl_interop.h>
#include "raytracer/core/cuda_utils.cuh"
#include "raytracer/core/vec3.cuh"

namespace rt {

// Kernel to convert HDR Color buffer to RGBA8 for display
__global__ void convert_to_rgba8(
    uchar4* output,
    const Color* hdr_buffer,
    int width, int height,
    int accumulated_samples
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height) return;

    int idx = j * width + i;

    // Get accumulated color and normalize by sample count
    Color c = hdr_buffer[idx];
    if (accumulated_samples > 0) {
        c = c / static_cast<float>(accumulated_samples);
    }

    // Tone mapping (Reinhard)
    c = Color(
        c.x / (1.0f + c.x),
        c.y / (1.0f + c.y),
        c.z / (1.0f + c.z)
    );

    // Gamma correction
    constexpr float inv_gamma = 1.0f / 2.2f;
    c = Color(
        powf(fmaxf(0.0f, c.x), inv_gamma),
        powf(fmaxf(0.0f, c.y), inv_gamma),
        powf(fmaxf(0.0f, c.z), inv_gamma)
    );

    // Convert to 8-bit
    output[idx] = make_uchar4(
        static_cast<unsigned char>(fminf(255.0f, c.x * 255.999f)),
        static_cast<unsigned char>(fminf(255.0f, c.y * 255.999f)),
        static_cast<unsigned char>(fminf(255.0f, c.z * 255.999f)),
        255
    );
}

class GLInterop {
public:
    GLuint pbo;                        // Pixel Buffer Object
    GLuint texture;                    // Display texture
    cudaGraphicsResource* cuda_pbo;    // CUDA resource handle
    int width, height;

    GLInterop() : pbo(0), texture(0), cuda_pbo(nullptr), width(0), height(0) {}

    bool initialize(int w, int h) {
        width = w;
        height = h;

        // Create PBO
        glGenBuffers(1, &pbo);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBufferData(GL_PIXEL_UNPACK_BUFFER,
                     width * height * sizeof(uchar4),
                     nullptr,
                     GL_DYNAMIC_DRAW);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // Register PBO with CUDA
        cudaError_t err = cudaGraphicsGLRegisterBuffer(
            &cuda_pbo, pbo,
            cudaGraphicsMapFlagsWriteDiscard
        );
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to register PBO with CUDA: %s\n",
                    cudaGetErrorString(err));
            return false;
        }

        // Create display texture
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,
                     width, height, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);

        return true;
    }

    // Map PBO for CUDA writing - returns device pointer
    uchar4* map_for_cuda() {
        cudaGraphicsMapResources(1, &cuda_pbo, 0);

        uchar4* d_ptr;
        size_t size;
        cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&d_ptr), &size, cuda_pbo);

        return d_ptr;
    }

    // Unmap PBO after CUDA is done
    void unmap_from_cuda() {
        cudaGraphicsUnmapResources(1, &cuda_pbo, 0);
    }

    // Update texture from PBO and render fullscreen quad
    void display() {
        // Copy PBO to texture
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                       width, height,
                       GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // Render fullscreen quad
        glEnable(GL_TEXTURE_2D);
        glBegin(GL_QUADS);
            glTexCoord2f(0, 1); glVertex2f(-1, -1);
            glTexCoord2f(1, 1); glVertex2f( 1, -1);
            glTexCoord2f(1, 0); glVertex2f( 1,  1);
            glTexCoord2f(0, 0); glVertex2f(-1,  1);
        glEnd();
        glDisable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    void cleanup() {
        if (cuda_pbo) {
            cudaGraphicsUnregisterResource(cuda_pbo);
            cuda_pbo = nullptr;
        }
        if (pbo) {
            glDeleteBuffers(1, &pbo);
            pbo = 0;
        }
        if (texture) {
            glDeleteTextures(1, &texture);
            texture = 0;
        }
    }

    ~GLInterop() {
        // Note: cleanup() should be called explicitly before GL context destruction
    }
};

} // namespace rt

#endif // ENABLE_INTERACTIVE
#endif // RAYTRACER_INTERACTIVE_GL_INTEROP_CUH
