#ifndef RAYTRACER_CORE_INTERVAL_CUH
#define RAYTRACER_CORE_INTERVAL_CUH

/**
 * @file interval.cuh
 * @brief Definition de la classe Interval pour gerer des plages de valeurs [min, max]
 * @details Les intervalles sont utilises partout dans le raytracer : pour definir
 *          les bornes de validite du parametre t lors de l'intersection des rayons,
 *          pour representer les etendues des boites englobantes (AABB) sur chaque axe,
 *          et pour clamper les valeurs de couleur avant l'affichage.
 */

#include <cfloat>
#include "raytracer/core/cuda_compat.cuh"

namespace rt {

/**
 * @class Interval
 * @brief Represente un intervalle ferme [min, max] de nombres flottants
 * @details Cette classe offre des operations courantes sur les intervalles :
 *          test d'appartenance, clamping, union de deux intervalles, etc.
 *          Par defaut, un intervalle est vide (min = +infini, max = -infini).
 */
class Interval {
public:
    float min; /**< Borne inferieure de l'intervalle */
    float max; /**< Borne superieure de l'intervalle */

    /**
     * @brief Constructeur par defaut, cree un intervalle vide
     * @details Un intervalle vide a min = +FLT_MAX et max = -FLT_MAX,
     *          ce qui signifie qu'il ne contient aucune valeur.
     */
    __host__ __device__ Interval() : min(+FLT_MAX), max(-FLT_MAX) {}

    /**
     * @brief Constructeur a partir de bornes explicites
     * @param _min La borne inferieure
     * @param _max La borne superieure
     */
    __host__ __device__ Interval(float _min, float _max) : min(_min), max(_max) {}

    /**
     * @brief Constructeur d'union de deux intervalles
     * @details Cree le plus petit intervalle contenant a la fois a et b.
     *          Utile pour fusionner des boites englobantes.
     * @param a Premier intervalle
     * @param b Deuxieme intervalle
     */
    __host__ __device__ Interval(const Interval& a, const Interval& b) {
        min = a.min <= b.min ? a.min : b.min;
        max = a.max >= b.max ? a.max : b.max;
    }

    /**
     * @brief Calcule la taille (etendue) de l'intervalle
     * @return La difference max - min
     */
    __host__ __device__ float size() const {
        return max - min;
    }

    /**
     * @brief Verifie si une valeur est contenue dans l'intervalle (bornes incluses)
     * @param x La valeur a tester
     * @return true si min <= x <= max
     */
    __host__ __device__ bool contains(float x) const {
        return min <= x && x <= max;
    }

    /**
     * @brief Verifie si une valeur est strictement a l'interieur de l'intervalle
     * @details Contrairement a contains(), les bornes sont exclues.
     *          Utilise pour les tests d'intersection ou l'on veut exclure
     *          les points exactement sur les bords.
     * @param x La valeur a tester
     * @return true si min < x < max
     */
    __host__ __device__ bool surrounds(float x) const {
        return min < x && x < max;
    }

    /**
     * @brief Restreint une valeur a l'intervalle [min, max]
     * @details Si x < min, retourne min. Si x > max, retourne max.
     *          Sinon retourne x. Tres utile pour clamper les couleurs
     *          avant la conversion en entier 8 bits.
     * @param x La valeur a clamper
     * @return La valeur clampee dans [min, max]
     */
    __host__ __device__ float clamp(float x) const {
        if (x < min) return min;
        if (x > max) return max;
        return x;
    }

    /**
     * @brief Retourne un intervalle agrandi symetriquement
     * @details Ajoute delta/2 de chaque cote de l'intervalle.
     *          Utilise pour ajouter un peu d'epaisseur aux boites
     *          englobantes trop fines (ex: triangles alignes sur un axe).
     * @param delta L'epaisseur totale a ajouter
     * @return Le nouvel intervalle elargi
     */
    __host__ __device__ Interval expand(float delta) const {
        float padding = delta / 2.0f;
        return Interval(min - padding, max + padding);
    }

};


}

#endif
