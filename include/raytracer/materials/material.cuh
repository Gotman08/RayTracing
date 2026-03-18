#ifndef RAYTRACER_MATERIALS_MATERIAL_CUH
#define RAYTRACER_MATERIALS_MATERIAL_CUH

/**
 * @file material.cuh
 * @brief Definition du materiau generique utilise dans le raytracer.
 * @details Ce fichier contient l'enumeration des types de materiaux disponibles
 *          ainsi que la classe Material qui encapsule toutes les proprietes
 *          necessaires pour decrire l'apparence d'un objet dans la scene.
 */

#include "raytracer/core/vec3.cuh"
#include "raytracer/core/ray.cuh"

namespace rt {

/**
 * @enum MaterialType
 * @brief Enumeration des differents types de materiaux supportes par le raytracer.
 * @details Chaque type correspond a un modele d'interaction lumiere-surface different :
 *          - LAMBERTIAN : surface diffuse (mat), la lumiere est renvoyee dans toutes les directions.
 *          - METAL : surface metallique reflechissante, avec possibilite de flou (fuzz).
 *          - DIELECTRIC : materiau transparent comme le verre, capable de refraction et reflexion.
 */
enum class MaterialType {
    LAMBERTIAN,
    METAL,
    DIELECTRIC
};

/**
 * @class Material
 * @brief Represente un materiau generique attache a un objet de la scene.
 * @details Cette classe stocke toutes les proprietes physiques d'un materiau :
 *          le type (lambertien, metal ou dielectrique), la couleur (albedo),
 *          le facteur de flou pour les metaux (fuzz) et l'indice de refraction
 *          pour les dielectriques (ior). Elle est utilisable a la fois sur le
 *          CPU (host) et le GPU (device) grace aux qualificateurs CUDA.
 */
class Material {
public:
    MaterialType type;   ///< Type du materiau (lambertien, metal ou dielectrique)
    Color albedo;        ///< Couleur de base du materiau (proportion de lumiere reflechie par canal RGB)
    float fuzz;          ///< Facteur de flou pour les metaux (0 = miroir parfait, 1 = tres flou)
    float ior;           ///< Indice de refraction pour les dielectriques (ex: 1.5 pour le verre)

    /**
     * @brief Constructeur par defaut.
     * @details Initialise un materiau lambertien gris (albedo 0.5) sans flou
     *          et avec un indice de refraction de 1.5 par defaut.
     */
    __host__ __device__ Material()
        : type(MaterialType::LAMBERTIAN), albedo(0.5f, 0.5f, 0.5f),
          fuzz(0), ior(1.5f) {}

    /**
     * @brief Constructeur avec type et couleur.
     * @details Cree un materiau du type specifie avec la couleur donnee.
     *          Le fuzz est mis a 0 et l'ior a 1.5 par defaut.
     * @param t Le type du materiau (LAMBERTIAN, METAL ou DIELECTRIC).
     * @param a La couleur (albedo) du materiau.
     */
    __host__ __device__ Material(MaterialType t, const Color& a)
        : type(t), albedo(a), fuzz(0), ior(1.5f) {}

    /**
     * @brief Constructeur avec type, couleur et facteur de flou.
     * @details Cree un materiau avec tous les parametres principaux.
     *          Utile principalement pour les materiaux metalliques ou le fuzz
     *          controle la nettete de la reflexion.
     * @param t Le type du materiau.
     * @param a La couleur (albedo) du materiau.
     * @param f Le facteur de flou (perturbation de la reflexion).
     */
    __host__ __device__ Material(MaterialType t, const Color& a, float f)
        : type(t), albedo(a), fuzz(f), ior(1.5f) {}

    /**
     * @brief Fabrique statique pour creer un materiau dielectrique.
     * @details Methode utilitaire qui configure correctement un materiau
     *          de type DIELECTRIC avec un albedo blanc (le verre ne colore
     *          pas la lumiere) et l'indice de refraction specifie.
     * @param index_of_refraction L'indice de refraction du milieu (ex: 1.5 pour le verre).
     * @return Un objet Material configure comme dielectrique.
     */
    __host__ __device__ static Material make_dielectric(float index_of_refraction) {
        Material m;
        m.type = MaterialType::DIELECTRIC;
        m.albedo = Color(1, 1, 1);
        m.ior = index_of_refraction;
        return m;
    }

    /**
     * @brief Retourne la couleur (albedo) du materiau au point donne.
     * @details Pour l'instant, cette methode retourne simplement l'albedo
     *          uniforme du materiau, mais elle pourrait etre etendue pour
     *          supporter des textures en utilisant les coordonnees UV.
     * @param u Coordonnee de texture U (non utilisee pour l'instant).
     * @param v Coordonnee de texture V (non utilisee pour l'instant).
     * @param p Le point 3D sur la surface de l'objet.
     * @return La couleur du materiau au point donne.
     */
    __host__ __device__ Color get_albedo(float u, float v, const Point3& p) const {
        return albedo;
    }

    /**
     * @brief Retourne la lumiere emise par le materiau.
     * @details Par defaut, un materiau n'emet pas de lumiere (retourne noir).
     *          Cette methode peut etre surchargee pour creer des materiaux
     *          emissifs (sources de lumiere dans la scene).
     * @param u Coordonnee de texture U.
     * @param v Coordonnee de texture V.
     * @param p Le point 3D sur la surface de l'objet.
     * @return La couleur emise (noir par defaut, soit aucune emission).
     */
    __host__ __device__ Color emitted(float u, float v, const Point3& p) const {
        return Color(0, 0, 0);
    }
};

}

#endif
