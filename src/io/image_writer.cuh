#ifndef RAYTRACER_IO_IMAGE_WRITER_CUH
#define RAYTRACER_IO_IMAGE_WRITER_CUH

#include <string>
#include <vector>
#include <fstream>
#include <omp.h>
#include "raytracer/core/vec3.cuh"

namespace rt {

inline void save_image(const std::string& filename, Color* buffer, int width, int height) {
    std::vector<unsigned char> pixels(width * height * 3);

    // OPTIMIZED: Parallel pixel conversion
    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < width * height; idx++) {
        pixels[idx * 3 + 0] = static_cast<unsigned char>(255.999f * buffer[idx].x);
        pixels[idx * 3 + 1] = static_cast<unsigned char>(255.999f * buffer[idx].y);
        pixels[idx * 3 + 2] = static_cast<unsigned char>(255.999f * buffer[idx].z);
    }

    std::string ext = filename.substr(filename.find_last_of('.') + 1);

    if (ext == "png") {
        stbi_write_png(filename.c_str(), width, height, 3, pixels.data(), width * 3);
    } else if (ext == "jpg" || ext == "jpeg") {
        stbi_write_jpg(filename.c_str(), width, height, 3, pixels.data(), 95);
    } else if (ext == "bmp") {
        // FAST: BMP has no compression
        stbi_write_bmp(filename.c_str(), width, height, 3, pixels.data());
    } else if (ext == "tga") {
        // FAST: TGA with no compression
        stbi_write_tga(filename.c_str(), width, height, 3, pixels.data());
    } else {
        // PPM format (text, slow but simple)
        std::ofstream file(filename);
        file << "P3\n" << width << " " << height << "\n255\n";
        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) {
                int idx = j * width + i;
                file << static_cast<int>(255.999f * buffer[idx].x) << " "
                     << static_cast<int>(255.999f * buffer[idx].y) << " "
                     << static_cast<int>(255.999f * buffer[idx].z) << "\n";
            }
        }
    }
}

}

#endif
