#ifndef RAYTRACER_ACCELERATION_BVH_CUH
#define RAYTRACER_ACCELERATION_BVH_CUH

/**
 * @file bvh.cuh
 * @brief Structure d'acceleration BVH (Bounding Volume Hierarchy)
 * @details Ce fichier definit le noeud BVH et la classe BVH qui permet d'accelerer
 *          le calcul d'intersections rayon-scene en organisant les primitives dans
 *          un arbre de boites englobantes. La traversee est iterative (avec pile)
 *          pour etre compatible GPU.
 */

#include "raytracer/core/aabb.cuh"
#include "raytracer/core/ray.cuh"
#include "raytracer/geometry/hittable_list.cuh"

namespace rt {

/**
 * @brief Noeud de l'arbre BVH (Bounding Volume Hierarchy)
 * @details Chaque noeud contient une boite englobante (AABB) qui enveloppe tous
 *          ses enfants. Un noeud est soit interne (avec deux enfants left/right),
 *          soit une feuille (avec un indice vers une primitive).
 */
struct BVHNode {
    AABB bounds;         ///< Boite englobante alignee sur les axes (AABB) du noeud
    int left;            ///< Indice du fils gauche dans le tableau de noeuds (-1 si aucun)
    int right;           ///< Indice du fils droit dans le tableau de noeuds (-1 si aucun)
    int primitive_idx;   ///< Indice de la primitive dans le tableau (valide seulement si is_leaf)
    bool is_leaf;        ///< Vrai si le noeud est une feuille contenant une primitive

    __host__ __device__ BVHNode()
        : left(-1), right(-1), primitive_idx(-1), is_leaf(false) {}
};

/**
 * @brief Structure d'acceleration par hierarchie de volumes englobants
 * @details Le BVH organise les primitives de la scene dans un arbre binaire de
 *          boites englobantes. Cela permet de rejeter rapidement les rayons qui
 *          ne touchent pas un groupe de primitives, ce qui accelere enormement
 *          le rendu par rapport a un test d'intersection avec chaque primitive.
 *          La structure est stockee sous forme de tableau plat (pas de pointeurs)
 *          pour etre compatible avec la memoire GPU.
 */
class BVH {
public:
    BVHNode* nodes;          ///< Tableau de noeuds du BVH (memoire GPU ou CPU)
    HittableObject* primitives;  ///< Tableau des primitives de la scene
    int num_nodes;           ///< Nombre total de noeuds dans l'arbre
    int num_primitives;      ///< Nombre total de primitives

    __host__ __device__ BVH()
        : nodes(nullptr), primitives(nullptr), num_nodes(0), num_primitives(0) {}

    /**
     * @brief Traversee iterative du BVH pour trouver l'intersection la plus proche
     * @details Utilise une pile locale (tableau de taille fixe 64) au lieu de la
     *          recursion pour etre compatible GPU. Parcourt l'arbre en profondeur
     *          et teste l'intersection avec les boites englobantes avant de tester
     *          les primitives, ce qui permet d'eliminer rapidement les branches inutiles.
     * @param r Le rayon a tester
     * @param ray_t Intervalle de distance valide pour l'intersection [tmin, tmax]
     * @param rec Enregistrement d'intersection rempli si un hit est trouve
     * @return true si le rayon intersecte au moins une primitive, false sinon
     */
    __host__ __device__ bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const {
        if (num_nodes == 0) return false;

        bool hit_anything = false;
        float closest_so_far = ray_t.max;

        int stack[64];
        int stack_ptr = 0;
        stack[stack_ptr++] = 0;

        while (stack_ptr > 0) {
            int node_idx = stack[--stack_ptr];

            const BVHNode& node = nodes[node_idx];

            if (!node.bounds.hit(r, Interval(ray_t.min, closest_so_far)))
                continue;

            if (node.is_leaf) {
                HitRecord temp_rec;
                if (primitives[node.primitive_idx].hit(r, Interval(ray_t.min, closest_so_far), temp_rec)) {
                    hit_anything = true;
                    closest_so_far = temp_rec.t;
                    rec = temp_rec;
                }
            } else {
                if (node.right >= 0) stack[stack_ptr++] = node.right;
                if (node.left >= 0) stack[stack_ptr++] = node.left;
            }
        }

        return hit_anything;
    }

    /**
     * @brief Retourne la boite englobante de la racine du BVH
     * @details La boite englobante de la racine contient toute la scene.
     *          Si le BVH est vide, retourne une AABB par defaut.
     * @return La boite englobante AABB de l'ensemble de la scene
     */
    __host__ __device__ AABB bounding_box() const {
        if (num_nodes > 0)
            return nodes[0].bounds;
        return AABB();
    }
};

}

#endif
