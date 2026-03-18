#ifndef RAYTRACER_ENVIRONMENT_SKY_CUH
#define RAYTRACER_ENVIRONMENT_SKY_CUH

/**
 * @file sky.cuh
 * @brief Modele de ciel avec gradient et soleil optionnel
 * @details Ce fichier definit la classe Sky pour simuler un ciel avec un degrade
 *          entre l'horizon et le zenith, ainsi qu'un disque solaire optionnel.
 *          Deux fonctions libres sont aussi fournies : sky_gradient() pour un
 *          ciel bleu classique et sky_black() pour un fond noir.
 */

#include "raytracer/core/vec3.cuh"

namespace rt {

/**
 * @brief Modele de ciel avec gradient horizon/zenith et soleil optionnel
 * @details Le ciel est modelise par un degrade lineaire entre la couleur de
 *          l'horizon (en bas) et la couleur du zenith (en haut), base sur la
 *          composante Y de la direction du rayon. Un soleil peut etre ajoute
 *          optionnellement avec une direction, une intensite et une taille configurables.
 */
class Sky {
public:
    Color horizon_color;   ///< Couleur de l'horizon (bas du ciel)
    Color zenith_color;    ///< Couleur du zenith (haut du ciel)
    float sun_intensity;   ///< Intensite du soleil (0 = pas de soleil)
    Vec3 sun_direction;    ///< Direction normalisee vers le soleil
    float sun_size;        ///< Taille angulaire du disque solaire (en fraction)

    __host__ __device__ Sky()
        : horizon_color(1.0f, 1.0f, 1.0f),
          zenith_color(0.5f, 0.7f, 1.0f),
          sun_intensity(0.0f),
          sun_direction(0, 1, 0),
          sun_size(0.01f) {}

    /**
     * @brief Constructeur avec couleurs personnalisees pour l'horizon et le zenith
     * @param horizon Couleur de l'horizon
     * @param zenith Couleur du zenith
     */
    __host__ __device__ Sky(const Color& horizon, const Color& zenith)
        : horizon_color(horizon), zenith_color(zenith),
          sun_intensity(0.0f), sun_direction(0, 1, 0), sun_size(0.01f) {}

    /**
     * @brief Configure la direction et l'intensite du soleil
     * @details La direction est automatiquement normalisee. Le soleil apparait
     *          comme un disque lumineux dans le ciel lorsque l'intensite est > 0.
     * @param dir Direction vers le soleil (sera normalisee)
     * @param intensity Intensite lumineuse du soleil
     * @param size Taille angulaire du disque solaire (defaut : 0.01)
     */
    __host__ __device__ void set_sun(const Vec3& dir, float intensity, float size = 0.01f) {
        sun_direction = unit_vector(dir);
        sun_intensity = intensity;
        sun_size = size;
    }

    /**
     * @brief Calcule la couleur du ciel pour une direction de rayon donnee
     * @details Interpole lineairement entre la couleur de l'horizon et du zenith
     *          selon la composante Y de la direction normalisee. Si le soleil est
     *          actif, ajoute une contribution lumineuse lorsque la direction du rayon
     *          est proche de la direction du soleil.
     * @param direction Direction du rayon pour laquelle calculer la couleur du ciel
     * @return La couleur du ciel dans cette direction
     */
    __host__ __device__ Color get_color(const Vec3& direction) const {
        Vec3 unit_dir = unit_vector(direction);

        float t = 0.5f * (unit_dir.y + 1.0f);
        Color sky = (1.0f - t) * horizon_color + t * zenith_color;

        if (sun_intensity > 0.0f) {
            float sun_dot = dot(unit_dir, sun_direction);
            if (sun_dot > (1.0f - sun_size)) {
                float sun_factor = (sun_dot - (1.0f - sun_size)) / sun_size;
                sun_factor = sun_factor * sun_factor;
                sky = sky + sun_intensity * sun_factor * Color(1.0f, 0.95f, 0.8f);
            }
        }

        return sky;
    }
};

/**
 * @brief Fonction libre pour un ciel bleu avec gradient classique
 * @details Retourne un degrade allant du blanc (horizon) au bleu clair (zenith).
 *          C'est le ciel par defaut utilise dans beaucoup de raytracers.
 * @param direction Direction du rayon
 * @return Couleur du ciel interpolee entre blanc et bleu
 */
__host__ __device__ inline Color sky_gradient(const Vec3& direction) {
    Vec3 unit_dir = unit_vector(direction);
    float t = 0.5f * (unit_dir.y + 1.0f);
    return (1.0f - t) * Color(1.0f, 1.0f, 1.0f) + t * Color(0.5f, 0.7f, 1.0f);
}

/**
 * @brief Fonction libre pour un fond completement noir
 * @details Utile pour les scenes ou l'on ne veut aucune lumiere ambiante du ciel,
 *          par exemple les scenes d'interieur ou les scenes avec eclairage artificiel.
 * @param direction Direction du rayon (ignoree)
 * @return Noir (0, 0, 0)
 */
__host__ __device__ inline Color sky_black(const Vec3& direction) {
    return Color(0.0f, 0.0f, 0.0f);
}

}

#endif
