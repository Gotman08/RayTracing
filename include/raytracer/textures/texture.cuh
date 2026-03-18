#ifndef RAYTRACER_TEXTURES_TEXTURE_CUH
#define RAYTRACER_TEXTURES_TEXTURE_CUH

/**
 * @file texture.cuh
 * @brief Classe de base pour les textures
 * @details Ce fichier definit la classe Texture qui represente une texture simple
 *          (couleur solide). Elle sert de classe de base pour le systeme de textures
 *          du raytracer.
 */

#include "raytracer/core/vec3.cuh"

namespace rt {

/**
 * @brief Texture de base (couleur solide)
 * @details Cette classe represente la texture la plus simple : une couleur constante.
 *          Elle peut etre utilisee directement ou servir de base pour des textures
 *          plus complexes (damier, bruit de Perlin, image, etc.).
 */
class Texture {
public:
    Color color;  ///< Couleur de la texture (blanc par defaut)

    __host__ __device__ Texture() : color(1, 1, 1) {}

    /**
     * @brief Constructeur avec une couleur donnee
     * @param c La couleur de la texture
     */
    __host__ __device__ Texture(const Color& c) : color(c) {}

    /**
     * @brief Retourne la couleur de la texture aux coordonnees donnees
     * @details Pour une texture solide, la couleur est constante et les parametres
     *          u, v et p sont ignores. Dans des sous-classes, ces parametres
     *          permettraient de varier la couleur selon la position.
     * @param u Coordonnee de texture horizontale (0 a 1)
     * @param v Coordonnee de texture verticale (0 a 1)
     * @param p Point d'intersection dans l'espace monde
     * @return La couleur de la texture a cet endroit
     */
    __host__ __device__ Color value(float u, float v, const Point3& p) const {
        return color;
    }
};

}

#endif
