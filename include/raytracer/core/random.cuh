#ifndef RAYTRACER_CORE_RANDOM_CUH
#define RAYTRACER_CORE_RANDOM_CUH

/**
 * @file random.cuh
 * @brief Generateur de nombres aleatoires pour le rendu CPU
 * @details Ce fichier fournit la classe CPURandom qui encapsule le generateur
 *          mt19937 de la bibliotheque standard, ainsi que des fonctions utilitaires
 *          pour generer des vecteurs aleatoires utilises dans le lancer de rayons
 *          (directions diffuses, echantillonnage de disque, hemisphere, etc.).
 */

#include <random>
#include <cstdint>
#include <cmath>

namespace rt {

class Vec3;

/**
 * @class CPURandom
 * @brief Generateur de nombres aleatoires pour le rendu sur CPU
 * @details Cette classe encapsule le generateur Mersenne Twister (mt19937)
 *          et une distribution uniforme sur [0, 1). Elle permet de generer
 *          des floats aleatoires de maniere reproductible grace a une graine (seed).
 *          On l'utilise principalement pour l'echantillonnage Monte Carlo lors
 *          du rendu sur CPU.
 */
class CPURandom {
public:
    /**
     * @brief Constructeur du generateur aleatoire
     * @details Initialise le generateur mt19937 avec la graine donnee
     *          et prepare une distribution uniforme sur [0, 1).
     * @param seed Graine pour initialiser le generateur (defaut : 0)
     */
    CPURandom(uint64_t seed = 0) : gen(seed), dist(0.0f, 1.0f) {}

    /**
     * @brief Operateur d'appel pour generer un nombre aleatoire
     * @details Permet d'utiliser l'objet comme une fonction : rng()
     * @return Un float aleatoire dans l'intervalle [0, 1)
     */
    float operator()() {
        return dist(gen);
    }

    /**
     * @brief Genere un nombre flottant aleatoire dans [0, 1)
     * @return Un float aleatoire uniformement distribue dans [0, 1)
     */
    float random_float() {
        return dist(gen);
    }

    /**
     * @brief Genere un nombre flottant aleatoire dans un intervalle donne
     * @param min Borne inferieure de l'intervalle
     * @param max Borne superieure de l'intervalle
     * @return Un float aleatoire uniformement distribue dans [min, max)
     */
    float random_float(float min, float max) {
        return min + (max - min) * dist(gen);
    }

private:
    std::mt19937 gen;                          ///< Generateur Mersenne Twister
    std::uniform_real_distribution<float> dist; ///< Distribution uniforme [0, 1)
};

}

#include "raytracer/core/vec3.cuh"

namespace rt {

constexpr float RANDOM_TWO_PI = 2.0f * 3.14159265358979323846f; ///< Constante 2*PI pour les calculs d'angles

/**
 * @brief Genere un vecteur 3D avec des composantes aleatoires dans [0, 1)
 * @param rng Reference vers le generateur aleatoire CPU
 * @return Un Vec3 dont chaque composante est un float aleatoire dans [0, 1)
 */
inline Vec3 random_vec3(CPURandom& rng) {
    return Vec3(rng(), rng(), rng());
}

/**
 * @brief Genere un vecteur 3D avec des composantes aleatoires dans [min, max)
 * @param min Borne inferieure pour chaque composante
 * @param max Borne superieure pour chaque composante
 * @param rng Reference vers le generateur aleatoire CPU
 * @return Un Vec3 dont chaque composante est dans [min, max)
 */
inline Vec3 random_vec3(float min, float max, CPURandom& rng) {
    float range = max - min;
    return Vec3(
        min + range * rng(),
        min + range * rng(),
        min + range * rng()
    );
}

/**
 * @brief Genere un vecteur unitaire aleatoire uniformement distribue sur la sphere
 * @details Utilise la methode des coordonnees spheriques avec z uniforme dans [-1, 1]
 *          et phi uniforme dans [0, 2*PI) pour obtenir une distribution uniforme
 *          sur la surface de la sphere unite.
 * @param rng Reference vers le generateur aleatoire CPU
 * @return Un Vec3 de norme 1 oriente aleatoirement
 */
inline Vec3 random_unit_vector(CPURandom& rng) {
    float z = 2.0f * rng() - 1.0f;
    float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    float phi = RANDOM_TWO_PI * rng();
    return Vec3(r * cosf(phi), r * sinf(phi), z);
}

/**
 * @brief Genere un point aleatoire a l'interieur de la sphere unite
 * @details Combine un vecteur unitaire aleatoire avec un rayon mis a l'echelle
 *          par la racine cubique d'un nombre aleatoire, ce qui assure une
 *          distribution volumique uniforme dans la sphere.
 * @param rng Reference vers le generateur aleatoire CPU
 * @return Un Vec3 de norme <= 1
 */
inline Vec3 random_in_unit_sphere(CPURandom& rng) {
    Vec3 dir = random_unit_vector(rng);
    float r = cbrtf(rng());
    return dir * r;
}

/**
 * @brief Genere un vecteur unitaire aleatoire dans l'hemisphere defini par une normale
 * @details Genere un vecteur aleatoire sur la sphere unite, puis le retourne
 *          s'il pointe dans la direction opposee a la normale. Utile pour
 *          l'echantillonnage de la diffusion lambertienne.
 * @param normal La normale de la surface definissant l'hemisphere
 * @param rng Reference vers le generateur aleatoire CPU
 * @return Un Vec3 unitaire dans le meme hemisphere que la normale
 */
inline Vec3 random_on_hemisphere(const Vec3& normal, CPURandom& rng) {
    Vec3 on_unit_sphere = random_unit_vector(rng);
    if (dot(on_unit_sphere, normal) > 0.0f)
        return on_unit_sphere;
    else
        return -on_unit_sphere;
}

/**
 * @brief Genere un point aleatoire dans le disque unite (plan XY)
 * @details Utilise un echantillonnage polaire avec r = sqrt(u) pour assurer
 *          une distribution surfacique uniforme dans le disque. La composante z
 *          est toujours 0. Principalement utilise pour simuler la profondeur
 *          de champ (defocus blur) de la camera.
 * @param rng Reference vers le generateur aleatoire CPU
 * @return Un Vec3 dans le plan z=0, de norme <= 1
 */
inline Vec3 random_in_unit_disk(CPURandom& rng) {
    float r = sqrtf(rng());
    float theta = RANDOM_TWO_PI * rng();
    return Vec3(r * cosf(theta), r * sinf(theta), 0.0f);
}

/**
 * @brief Genere une direction aleatoire selon une distribution cosinus
 * @details Produit une direction dans l'hemisphere superieur (z > 0)
 *          avec une probabilite proportionnelle au cosinus de l'angle
 *          avec la normale (axe z). C'est l'echantillonnage optimal pour
 *          les surfaces lambertiennes car il suit la loi de Lambert.
 * @param rng Reference vers le generateur aleatoire CPU
 * @return Un Vec3 direction dans l'hemisphere superieur, pondere par le cosinus
 */
inline Vec3 random_cosine_direction(CPURandom& rng) {
    float r1 = rng();
    float r2 = rng();

    float phi = RANDOM_TWO_PI * r1;
    float x = cosf(phi) * sqrtf(r2);
    float y = sinf(phi) * sqrtf(r2);
    float z = sqrtf(1.0f - r2);

    return Vec3(x, y, z);
}

}

#endif
