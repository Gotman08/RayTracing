#ifndef RAYTRACER_TEXTURES_SOLID_COLOR_CUH
#define RAYTRACER_TEXTURES_SOLID_COLOR_CUH

/**
 * @file solid_color.cuh
 * @brief Texture de couleur unie (solide)
 * @details Ce fichier definit la classe SolidColor qui represente une texture
 *          retournant toujours la meme couleur (albedo), independamment des
 *          coordonnees UV ou de la position dans l'espace.
 */

#include "raytracer/core/vec3.cuh"

namespace rt {

/**
 * @brief Texture de couleur unie
 * @details Cette classe represente une texture qui retourne toujours la meme couleur
 *          (l'albedo). C'est utile pour les materiaux de couleur uniforme comme
 *          un mur peint ou une balle de couleur. L'albedo represente la fraction
 *          de lumiere reflechie par la surface.
 */
class SolidColor {
public:
    Color albedo;  ///< Couleur constante de la texture (noir par defaut)

    __host__ __device__ SolidColor() : albedo(0, 0, 0) {}

    /**
     * @brief Constructeur a partir d'un objet Color
     * @param c La couleur a utiliser comme albedo
     */
    __host__ __device__ SolidColor(const Color& c) : albedo(c) {}

    /**
     * @brief Constructeur a partir de composantes RGB
     * @param r Composante rouge (0 a 1)
     * @param g Composante verte (0 a 1)
     * @param b Composante bleue (0 a 1)
     */
    __host__ __device__ SolidColor(float r, float g, float b) : albedo(r, g, b) {}

    /**
     * @brief Retourne l'albedo constant de la texture
     * @details Les parametres u, v et p sont ignores car la couleur est uniforme
     *          sur toute la surface.
     * @param u Coordonnee de texture horizontale (ignoree)
     * @param v Coordonnee de texture verticale (ignoree)
     * @param p Point d'intersection dans l'espace monde (ignore)
     * @return La couleur albedo constante
     */
    __host__ __device__ Color value(float u, float v, const Point3& p) const {
        return albedo;
    }
};

}

#endif
