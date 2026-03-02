#ifndef RAYTRACER_ACCELERATION_BVH_CUH
#define RAYTRACER_ACCELERATION_BVH_CUH

#include "raytracer/core/aabb.cuh"
#include "raytracer/core/ray.cuh"
#include "raytracer/geometry/hittable_list.cuh"

namespace rt {

// Linear BVH node for GPU traversal
struct BVHNode {
    AABB bounds;
    int left;           // Left child index or first primitive (-1 if leaf)
    int right;          // Right child index or primitive count
    int primitive_idx;  // -1 for internal nodes
    bool is_leaf;

    __host__ __device__ BVHNode()
        : left(-1), right(-1), primitive_idx(-1), is_leaf(false) {}
};

class BVH {
public:
    BVHNode* nodes;
    HittableObject* primitives;
    int num_nodes;
    int num_primitives;

    __host__ __device__ BVH()
        : nodes(nullptr), primitives(nullptr), num_nodes(0), num_primitives(0) {}

    // Iterative BVH traversal for GPU
    __device__ bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const {
        if (num_nodes == 0) return false;

        bool hit_anything = false;
        float closest_so_far = ray_t.max;

        // Stack for iterative traversal
        int stack[64];
        int stack_ptr = 0;
        stack[stack_ptr++] = 0; // Start with root

        while (stack_ptr > 0) {
            int node_idx = stack[--stack_ptr];
            const BVHNode& node = nodes[node_idx];

            // Test AABB
            if (!node.bounds.hit(r, Interval(ray_t.min, closest_so_far)))
                continue;

            if (node.is_leaf) {
                // Test primitive
                HitRecord temp_rec;
                if (primitives[node.primitive_idx].hit(r, Interval(ray_t.min, closest_so_far), temp_rec)) {
                    hit_anything = true;
                    closest_so_far = temp_rec.t;
                    rec = temp_rec;
                }
            } else {
                // Push children (can optimize by ordering by distance)
                if (node.left >= 0) stack[stack_ptr++] = node.left;
                if (node.right >= 0) stack[stack_ptr++] = node.right;
            }
        }

        return hit_anything;
    }

    __host__ __device__ const AABB& bounding_box() const {
        if (num_nodes > 0)
            return nodes[0].bounds;
        static AABB empty;
        return empty;
    }
};

} // namespace rt

#endif // RAYTRACER_ACCELERATION_BVH_CUH
