/**
 * @file image_writer.cuh
 * @brief Sauvegarde du frame buffer en fichier image
 * @details Ce fichier fournit une fonction utilitaire pour exporter le resultat
 *          du rendu (frame buffer) vers differents formats d'image : PNG, JPG,
 *          BMP, TGA (via stb_image_write) ou PPM (ecriture manuelle).
 *          La conversion des couleurs flottantes vers des octets est parallelisee
 *          avec OpenMP pour de meilleures performances.
 */

#ifndef RAYTRACER_IO_IMAGE_WRITER_CUH
#define RAYTRACER_IO_IMAGE_WRITER_CUH

#include <string>
#include <vector>
#include <fstream>
#include <omp.h>
#include "raytracer/core/vec3.cuh"

namespace rt {

/**
 * @brief Sauvegarde le frame buffer en fichier image dans le format determine par l'extension
 * @details Convertit chaque pixel du buffer (couleurs flottantes entre 0 et 1) en
 *          valeurs octets (0-255) de maniere parallelisee avec OpenMP. Le format
 *          de sortie est determine automatiquement par l'extension du fichier :
 *          .png, .jpg/.jpeg, .bmp, .tga utilisent la bibliotheque stb_image_write,
 *          et tout autre format produit un fichier PPM (format texte simple).
 * @param filename Chemin du fichier de sortie (l'extension determine le format)
 * @param buffer Pointeur vers le tableau de pixels Color (valeurs flottantes RGB)
 * @param width Largeur de l'image en pixels
 * @param height Hauteur de l'image en pixels
 */
inline void save_image(const std::string& filename, Color* buffer, int width, int height) {
    std::vector<unsigned char> pixels(width * height * 3);

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
        stbi_write_bmp(filename.c_str(), width, height, 3, pixels.data());
    } else if (ext == "tga") {
        stbi_write_tga(filename.c_str(), width, height, 3, pixels.data());
    } else {
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
