#ifndef RAYTRACER_ACCELERATION_BVH_BUILDER_CUH
#define RAYTRACER_ACCELERATION_BVH_BUILDER_CUH

/**
 * @file bvh_builder.cuh
 * @brief Construction du BVH sur CPU et transfert vers GPU/CPU
 * @details Ce fichier contient la classe BVHBuilder qui construit la hierarchie
 *          de volumes englobants (BVH) de facon recursive sur le CPU, puis permet
 *          de copier la structure resultante en memoire GPU (via CUDA) ou en
 *          memoire CPU pour le rendu.
 */

#include "raytracer/acceleration/bvh.cuh"
#include <vector>
#include <algorithm>

namespace rt {

/**
 * @brief Constructeur de BVH sur CPU
 * @details Cette classe construit l'arbre BVH de maniere recursive en triant
 *          les primitives selon l'axe le plus long de leur boite englobante,
 *          puis en coupant au milieu. Une fois construit, le BVH peut etre
 *          transfere en memoire GPU ou CPU pour etre utilise lors du rendu.
 */
class BVHBuilder {
public:
    std::vector<BVHNode> nodes;          ///< Tableau des noeuds construits
    std::vector<HittableObject> primitives;  ///< Tableau des primitives (triees lors de la construction)

    /**
     * @brief Lance la construction du BVH a partir d'un tableau d'objets
     * @details Copie les objets dans le vecteur interne, puis appelle la methode
     *          recursive build_recursive() pour construire l'arbre de bas en haut.
     *          Le tableau de noeuds est reserve a 2*count pour eviter les reallocations.
     * @param objects Tableau d'objets hittables a organiser dans le BVH
     * @param count Nombre d'objets dans le tableau
     */
    void build(HittableObject* objects, int count) {
        primitives.assign(objects, objects + count);
        nodes.clear();
        nodes.reserve(2 * count);

        if (count == 0) return;

        build_recursive(0, count);
    }

    /**
     * @brief Copie le BVH construit en memoire GPU (CUDA)
     * @details Alloue de la memoire sur le GPU avec cudaMalloc et copie les tableaux
     *          de noeuds et de primitives du CPU vers le GPU. Le BVH retourne est
     *          pret a etre utilise dans un kernel CUDA pour le rendu.
     * @return Un objet BVH dont les pointeurs nodes et primitives sont en memoire GPU
     */
    BVH create_gpu_bvh() {
        BVH bvh;
        bvh.num_nodes = static_cast<int>(nodes.size());
        bvh.num_primitives = static_cast<int>(primitives.size());

        CUDA_CHECK(cudaMalloc(&bvh.nodes, nodes.size() * sizeof(BVHNode)));
        CUDA_CHECK(cudaMalloc(&bvh.primitives, primitives.size() * sizeof(HittableObject)));

        CUDA_CHECK(cudaMemcpy(bvh.nodes, nodes.data(),
                              nodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(bvh.primitives, primitives.data(),
                              primitives.size() * sizeof(HittableObject), cudaMemcpyHostToDevice));

        return bvh;
    }

    /**
     * @brief Libere la memoire GPU occupee par le BVH
     * @details Appelle cudaFree sur les tableaux de noeuds et de primitives,
     *          puis met les pointeurs a nullptr pour eviter les double-free.
     * @param bvh Reference vers le BVH dont la memoire GPU doit etre liberee
     */
    void free_gpu_bvh(BVH& bvh) {
        if (bvh.nodes) cudaFree(bvh.nodes);
        if (bvh.primitives) cudaFree(bvh.primitives);
        bvh.nodes = nullptr;
        bvh.primitives = nullptr;
    }

    /**
     * @brief Copie le BVH construit en memoire CPU (allocation dynamique)
     * @details Alloue de la memoire sur le CPU avec new[] et copie les tableaux
     *          de noeuds et de primitives. Utile pour le rendu CPU sans CUDA.
     * @return Un objet BVH dont les pointeurs nodes et primitives sont en memoire CPU
     */
    BVH create_cpu_bvh() {
        BVH bvh;
        bvh.num_nodes = static_cast<int>(nodes.size());
        bvh.num_primitives = static_cast<int>(primitives.size());

        bvh.nodes = new BVHNode[nodes.size()];
        bvh.primitives = new HittableObject[primitives.size()];

        std::copy(nodes.begin(), nodes.end(), bvh.nodes);
        std::copy(primitives.begin(), primitives.end(), bvh.primitives);

        return bvh;
    }

    /**
     * @brief Libere la memoire CPU occupee par le BVH
     * @details Appelle delete[] sur les tableaux de noeuds et de primitives,
     *          puis met les pointeurs a nullptr.
     * @param bvh Reference vers le BVH dont la memoire CPU doit etre liberee
     */
    void free_cpu_bvh(BVH& bvh) {
        if (bvh.nodes) delete[] bvh.nodes;
        if (bvh.primitives) delete[] bvh.primitives;
        bvh.nodes = nullptr;
        bvh.primitives = nullptr;
    }

private:
    /**
     * @brief Construit l'arbre BVH de facon recursive
     * @details Pour le sous-ensemble de primitives [start, end), calcule la boite
     *          englobante globale, puis :
     *          - Si une seule primitive : cree une feuille
     *          - Sinon : trie les primitives selon l'axe le plus long de la boite
     *            englobante, coupe au milieu, et construit recursivement les deux
     *            sous-arbres (gauche et droit)
     * @param start Indice de debut dans le tableau de primitives (inclus)
     * @param end Indice de fin dans le tableau de primitives (exclus)
     * @return L'indice du noeud cree dans le tableau nodes
     */
    int build_recursive(int start, int end) {
        int node_idx = static_cast<int>(nodes.size());
        nodes.emplace_back();

        AABB bounds = primitives[start].bbox;
        for (int i = start + 1; i < end; i++) {
            bounds = AABB(bounds, primitives[i].bbox);
        }
        nodes[node_idx].bounds = bounds;

        int count = end - start;

        if (count == 1) {
            nodes[node_idx].is_leaf = true;
            nodes[node_idx].primitive_idx = start;
            return node_idx;
        }

        int axis = bounds.longest_axis();

        std::sort(primitives.begin() + start, primitives.begin() + end,
            [axis](const HittableObject& a, const HittableObject& b) {
                return a.bbox.centroid()[axis] < b.bbox.centroid()[axis];
            });

        int mid = start + count / 2;

        nodes[node_idx].left = build_recursive(start, mid);
        nodes[node_idx].right = build_recursive(mid, end);
        nodes[node_idx].is_leaf = false;

        return node_idx;
    }
};

}

#endif
