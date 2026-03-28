#ifndef RAYTRACER_ACCELERATION_BVH_BUILDER_CUH
#define RAYTRACER_ACCELERATION_BVH_BUILDER_CUH

/** @file bvh_builder.cuh
 *  @brief Construction recursive du BVH + transfert GPU/CPU */

#include "raytracer/acceleration/bvh.cuh"
#include <vector>
#include <algorithm>

namespace rt {

/** @brief Construit le BVH (recursif, tri axe le plus long) */
class BVHBuilder {
public:
    std::vector<BVHNode> nodes;
    std::vector<HittableObject> primitives;

    /** @brief Lance la construction depuis un tableau d'objets */
    void build(HittableObject* objects, int count) {
        primitives.assign(objects, objects + count);
        nodes.clear();
        nodes.reserve(2 * count);

        if (count == 0) return;

        build_recursive(0, count);
    }

    /** @brief Copie le BVH en mem GPU */
#ifdef __CUDACC__
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

    /** @brief Free mem GPU du BVH */
    void free_gpu_bvh(BVH& bvh) {
        if (bvh.nodes) cudaFree(bvh.nodes);
        if (bvh.primitives) cudaFree(bvh.primitives);
        bvh.nodes = nullptr;
        bvh.primitives = nullptr;
    }
#endif

    /** @brief Copie le BVH en mem CPU (new[]) */
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

    /** @brief Free mem CPU du BVH */
    void free_cpu_bvh(BVH& bvh) {
        if (bvh.nodes) delete[] bvh.nodes;
        if (bvh.primitives) delete[] bvh.primitives;
        bvh.nodes = nullptr;
        bvh.primitives = nullptr;
    }

private:
    /** @brief Recursion interne - subdivise [start, end) */
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
