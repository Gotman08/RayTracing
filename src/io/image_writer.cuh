/**
 * Image Output Functions
 */

#ifndef RAYTRACER_IO_IMAGE_WRITER_CUH
#define RAYTRACER_IO_IMAGE_WRITER_CUH

#include <string>
#include <vector>
#include <fstream>
#include "stb_image_write.h"
#include "raytracer/core/vec3.cuh"

namespace rt {

inline void save_image(const std::string& filename, Color* buffer, int width, int height) {
    std::vector<unsigned char> pixels(width * height * 3);

    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            int idx = j * width + i;
            pixels[idx * 3 + 0] = static_cast<unsigned char>(255.999f * buffer[idx].x);
            pixels[idx * 3 + 1] = static_cast<unsigned char>(255.999f * buffer[idx].y);
            pixels[idx * 3 + 2] = static_cast<unsigned char>(255.999f * buffer[idx].z);
        }
    }

    std::string ext = filename.substr(filename.find_last_of('.') + 1);

    if (ext == "png") {
        stbi_write_png(filename.c_str(), width, height, 3, pixels.data(), width * 3);
    } else if (ext == "jpg" || ext == "jpeg") {
        stbi_write_jpg(filename.c_str(), width, height, 3, pixels.data(), 95);
    } else {
        // Default to PPM
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

} // namespace rt

#endif // RAYTRACER_IO_IMAGE_WRITER_CUH
