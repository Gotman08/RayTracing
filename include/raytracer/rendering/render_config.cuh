#ifndef RAYTRACER_RENDERING_RENDER_CONFIG_CUH
#define RAYTRACER_RENDERING_RENDER_CONFIG_CUH

/**
 * @file render_config.cuh
 * @brief Structure de configuration pour le moteur de rendu
 * @details Ce fichier definit la structure RenderConfig qui regroupe tous les
 *          parametres necessaires au rendu : resolution de l'image, nombre de
 *          samples par pixel, profondeur maximale des rebonds, et configuration
 *          de l'environnement (ciel ou couleur de fond).
 */

#include "raytracer/core/vec3.cuh"
#include "raytracer/environment/sky.cuh"

namespace rt {

/**
 * @brief Structure contenant tous les parametres de configuration du rendu
 * @details Cette structure regroupe les parametres essentiels pour configurer
 *          le rendu : la resolution de l'image, le nombre d'echantillons par pixel
 *          (pour l'antialiasing), la profondeur maximale de recursion des rayons,
 *          et le choix entre un ciel degrade ou une couleur de fond uniforme.
 *          Les valeurs par defaut donnent un bon compromis qualite/performance.
 */
struct RenderConfig {
    int width;                ///< Largeur de l'image en pixels
    int height;               ///< Hauteur de l'image en pixels
    int samples_per_pixel;    ///< Nombre d'echantillons par pixel (antialiasing)
    int max_depth;            ///< Profondeur maximale des rebonds de rayons
    bool use_sky;             ///< Si vrai, utilise le ciel degrade ; sinon, la couleur de fond
    Sky sky;                  ///< Configuration du ciel degrade (couleurs haut/bas)
    Color background;         ///< Couleur de fond uniforme (utilisee si use_sky est faux)

    /**
     * @brief Constructeur par defaut avec des valeurs raisonnables
     * @details Initialise une image 800x600 avec 100 samples par pixel,
     *          une profondeur maximale de 50 rebonds, et un ciel degrade
     *          allant du blanc (horizon) au bleu clair (zenith).
     */
    RenderConfig()
        : width(800), height(600), samples_per_pixel(100), max_depth(50),
          use_sky(true), background(0, 0, 0) {
        sky = Sky(Color(1, 1, 1), Color(0.5f, 0.7f, 1.0f));
    }
};

}

#endif
