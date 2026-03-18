#ifndef RAYTRACER_CORE_AABB_CUH
#define RAYTRACER_CORE_AABB_CUH

/**
 * @file aabb.cuh
 * @brief Definition de la classe AABB (Axis-Aligned Bounding Box)
 * @details Les boites englobantes alignees aux axes sont des volumes simples
 *          utilises pour accelerer le ray tracing. Avant de tester l'intersection
 *          d'un rayon avec un objet complexe, on teste d'abord la boite englobante
 *          qui est beaucoup moins couteuse. Elles sont la base de la structure
 *          d'acceleration BVH (Bounding Volume Hierarchy).
 */

#include "raytracer/core/vec3.cuh"
#include "raytracer/core/ray.cuh"
#include "raytracer/core/interval.cuh"

namespace rt {

/**
 * @class AABB
 * @brief Boite englobante alignee aux axes (Axis-Aligned Bounding Box)
 * @details Une AABB est definie par trois intervalles (un par axe x, y, z)
 *          qui forment un parallelepipede rectangle aligne avec les axes du repere.
 *          Grace a cet alignement, le test d'intersection avec un rayon est tres
 *          rapide (methode des slabs). Utilisee dans le BVH pour organiser la
 *          scene en hierarchie de volumes.
 */
class AABB {
public:
    Interval x; /**< Etendue de la boite le long de l'axe X */
    Interval y; /**< Etendue de la boite le long de l'axe Y */
    Interval z; /**< Etendue de la boite le long de l'axe Z */

    /**
     * @brief Constructeur par defaut, cree une AABB vide
     */
    __host__ __device__ AABB() {}

    /**
     * @brief Constructeur a partir de trois intervalles (un par axe)
     * @details Applique un padding minimal pour eviter les boites degenerees.
     * @param x Intervalle sur l'axe X
     * @param y Intervalle sur l'axe Y
     * @param z Intervalle sur l'axe Z
     */
    __host__ __device__ AABB(const Interval& x, const Interval& y, const Interval& z)
        : x(x), y(y), z(z) {
        pad_to_minimums();
    }

    /**
     * @brief Constructeur a partir de deux points opposes de la boite
     * @details Les intervalles sont automatiquement ordonnes (min <= max)
     *          quel que soit l'ordre des points. Un padding minimal est applique.
     * @param a Premier coin de la boite
     * @param b Coin oppose de la boite
     */
    __host__ __device__ AABB(const Point3& a, const Point3& b) {
        x = (a.x <= b.x) ? Interval(a.x, b.x) : Interval(b.x, a.x);
        y = (a.y <= b.y) ? Interval(a.y, b.y) : Interval(b.y, a.y);
        z = (a.z <= b.z) ? Interval(a.z, b.z) : Interval(b.z, a.z);
        pad_to_minimums();
    }

    /**
     * @brief Constructeur d'union de deux AABB
     * @details Cree la plus petite AABB contenant les deux boites donnees.
     *          Utilise lors de la construction du BVH pour fusionner les
     *          boites englobantes des noeuds enfants.
     * @param box0 Premiere boite englobante
     * @param box1 Deuxieme boite englobante
     */
    __host__ __device__ AABB(const AABB& box0, const AABB& box1) {
        x = Interval(box0.x, box1.x);
        y = Interval(box0.y, box1.y);
        z = Interval(box0.z, box1.z);
    }

    /**
     * @brief Retourne l'intervalle correspondant a un axe donne
     * @param n Indice de l'axe (0 = x, 1 = y, 2 = z)
     * @return Reference constante vers l'intervalle de l'axe demande
     */
    __host__ __device__ const Interval& axis_interval(int n) const {
        if (n == 1) return y;
        if (n == 2) return z;
        return x;
    }

    /**
     * @brief Test d'intersection rayon-boite par la methode des slabs
     * @details Pour chaque axe, on calcule les parametres t d'entree et de sortie
     *          du rayon dans la tranche (slab) definie par l'intervalle de cet axe.
     *          L'intersection existe si et seulement si les intervalles de t se
     *          chevauchent sur les trois axes simultanement. C'est un algorithme
     *          tres efficace et adapte au GPU.
     * @param r Le rayon a tester
     * @param ray_t L'intervalle de validite pour le parametre t du rayon
     * @return true si le rayon intersecte la boite dans l'intervalle donne
     */
    __host__ __device__ bool hit(const Ray& r, Interval ray_t) const {
        const Point3& orig = r.origin();
        const Vec3& dir = r.direction();

        float tmin = ray_t.min;
        float tmax = ray_t.max;

        {
            float invD = 1.0f / dir.x;
            float t0 = (x.min - orig.x) * invD;
            float t1 = (x.max - orig.x) * invD;
            if (invD < 0.0f) { float tmp = t0; t0 = t1; t1 = tmp; }
            tmin = t0 > tmin ? t0 : tmin;
            tmax = t1 < tmax ? t1 : tmax;
            if (tmax <= tmin) return false;
        }

        {
            float invD = 1.0f / dir.y;
            float t0 = (y.min - orig.y) * invD;
            float t1 = (y.max - orig.y) * invD;
            if (invD < 0.0f) { float tmp = t0; t0 = t1; t1 = tmp; }
            tmin = t0 > tmin ? t0 : tmin;
            tmax = t1 < tmax ? t1 : tmax;
            if (tmax <= tmin) return false;
        }

        {
            float invD = 1.0f / dir.z;
            float t0 = (z.min - orig.z) * invD;
            float t1 = (z.max - orig.z) * invD;
            if (invD < 0.0f) { float tmp = t0; t0 = t1; t1 = tmp; }
            tmin = t0 > tmin ? t0 : tmin;
            tmax = t1 < tmax ? t1 : tmax;
            if (tmax <= tmin) return false;
        }

        return true;
    }

    /**
     * @brief Retourne l'indice de l'axe le plus long de la boite
     * @details Utile pour le BVH : on trie les objets selon l'axe le plus
     *          long pour obtenir une meilleure partition spatiale.
     * @return 0 pour x, 1 pour y, 2 pour z
     */
    __host__ __device__ int longest_axis() const {
        if (x.size() > y.size())
            return x.size() > z.size() ? 0 : 2;
        else
            return y.size() > z.size() ? 1 : 2;
    }

    /**
     * @brief Calcule le centre geometrique (centroide) de la boite
     * @details Le centroide est la moyenne des bornes sur chaque axe.
     *          Utilise dans le BVH pour trier les objets spatialement.
     * @return Le point central de la boite
     */
    __host__ __device__ Point3 centroid() const {
        return Point3(
            (x.min + x.max) * 0.5f,
            (y.min + y.max) * 0.5f,
            (z.min + z.max) * 0.5f
        );
    }

    /**
     * @brief Calcule l'aire de la surface de la boite
     * @details La formule est 2*(dx*dy + dy*dz + dz*dx), c'est-a-dire
     *          la somme des aires des 6 faces. Utilisee dans l'heuristique
     *          SAH (Surface Area Heuristic) pour optimiser la construction du BVH.
     * @return L'aire totale de la surface de la boite
     */
    __host__ __device__ float surface_area() const {
        float dx = x.size();
        float dy = y.size();
        float dz = z.size();
        return 2.0f * (dx * dy + dy * dz + dz * dx);
    }

private:
    /**
     * @brief Ajoute un padding minimal aux axes trop fins
     * @details Si un axe a une etendue inferieure a 0.0001, on l'elargit
     *          pour eviter les boites degenerees (d'epaisseur nulle). Cela
     *          arrive par exemple avec des triangles parfaitement alignes
     *          sur un plan. Sans ce padding, le test d'intersection pourrait
     *          echouer a cause d'erreurs numeriques.
     */
    __host__ __device__ void pad_to_minimums() {
        float delta = 0.0001f;
        if (x.size() < delta) x = x.expand(delta);
        if (y.size() < delta) y = y.expand(delta);
        if (z.size() < delta) z = z.expand(delta);
    }
};

}

#endif
