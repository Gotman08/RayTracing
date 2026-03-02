#ifndef RAYTRACER_ACCELERATION_BVH_BUILDER_CUH
#define RAYTRACER_ACCELERATION_BVH_BUILDER_CUH

#include "raytracer/acceleration/bvh.cuh"
#include <vector>
#include <algorithm>

namespace rt {

class BVHBuilder {
public:
    std::vector<BVHNode> nodes;
    std::vector<HittableObject> primitives;

    void build(HittableObject* objects, int count) {
        primitives.assign(objects, objects + count);
        nodes.clear();
        nodes.reserve(2 * count);

        if (count == 0) return;

        build_recursive(0, count);
    }

    BVH create_gpu_bvh() {
        BVH bvh;
        bvh.num_nodes = static_cast<int>(nodes.size());
        bvh.num_primitives = static_cast<int>(primitives.size());

        // Allocate GPU memory
        CUDA_CHECK(cudaMalloc(&bvh.nodes, nodes.size() * sizeof(BVHNode)));
        CUDA_CHECK(cudaMalloc(&bvh.primitives, primitives.size() * sizeof(HittableObject)));

        // Copy to GPU
        CUDA_CHECK(cudaMemcpy(bvh.nodes, nodes.data(),
                              nodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(bvh.primitives, primitives.data(),
                              primitives.size() * sizeof(HittableObject), cudaMemcpyHostToDevice));

        return bvh;
    }

    void free_gpu_bvh(BVH& bvh) {
        if (bvh.nodes) cudaFree(bvh.nodes);
        if (bvh.primitives) cudaFree(bvh.primitives);
        bvh.nodes = nullptr;
        bvh.primitives = nullptr;
    }

private:
    int build_recursive(int start, int end) {
        int node_idx = static_cast<int>(nodes.size());
        nodes.emplace_back();

        // Compute bounds
        AABB bounds = primitives[start].bbox;
        for (int i = start + 1; i < end; i++) {
            bounds = AABB(bounds, primitives[i].bbox);
        }
        nodes[node_idx].bounds = bounds;

        int count = end - start;

        if (count == 1) {
            // Leaf node
            nodes[node_idx].is_leaf = true;
            nodes[node_idx].primitive_idx = start;
            return node_idx;
        }

        // Choose split axis (longest)
        int axis = bounds.longest_axis();

        // Sort primitives along axis
        std::sort(primitives.begin() + start, primitives.begin() + end,
            [axis](const HittableObject& a, const HittableObject& b) {
                return a.bbox.centroid()[axis] < b.bbox.centroid()[axis];
            });

        // Split at midpoint
        int mid = start + count / 2;

        // Build children
        nodes[node_idx].left = build_recursive(start, mid);
        nodes[node_idx].right = build_recursive(mid, end);
        nodes[node_idx].is_leaf = false;

        return node_idx;
    }
};

} // namespace rt

#endif // RAYTRACER_ACCELERATION_BVH_BUILDER_CUH
