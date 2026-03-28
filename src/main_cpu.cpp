/** @file main_cpu.cpp
 * @brief Entree du ray tracer CPU-only (pas de dep CUDA, BVH local) */

#include <iostream>
#include <chrono>
#include <ctime>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "raytracer/core/cuda_compat.cuh"
#include "raytracer/core/vec3.cuh"
#include "raytracer/core/ray.cuh"
#include "raytracer/core/random.cuh"
#include "raytracer/core/aabb.cuh"
#include "raytracer/core/interval.cuh"
#include "raytracer/core/hit_record.cuh"
#include "raytracer/geometry/hittable.cuh"
#include "raytracer/geometry/hittable_list.cuh"
#include "raytracer/geometry/sphere.cuh"
#include "raytracer/geometry/plane.cuh"
#include "raytracer/materials/material.cuh"
#include "raytracer/materials/lambertian.cuh"
#include "raytracer/materials/metal.cuh"
#include "raytracer/materials/dielectric.cuh"
#include "raytracer/camera/camera.cuh"
#include "raytracer/acceleration/bvh.cuh"
#include "raytracer/rendering/render_config.cuh"
#include "raytracer/rendering/material_dispatch.cuh"
#include "raytracer/rendering/tone_mapping.cuh"
#include "raytracer/rendering/cpu_renderer.cuh"

#include "cli/args.cuh"
#include "io/image_writer.cuh"
#include "scene/default_scene.cuh"

using namespace rt;

static constexpr int MAX_OBJECTS = 1000;
static constexpr int MAX_MATERIALS = 1000;

/** @brief BVH builder CPU-only (top-down, median split, sans CUDA) */
class CPUBVHBuilder {
public:
    std::vector<BVHNode> nodes;
    std::vector<HittableObject> primitives;

    /** @brief Construit le BVH a partir du tableau d'objets */
    void build(HittableObject* objects, int count) {
        primitives.assign(objects, objects + count);
        nodes.clear();
        nodes.reserve(2 * count);

        if (count == 0) return;

        build_recursive(0, count);
    }

    /** @brief Alloc + copie des noeuds/primitives dans un BVH autonome */
    BVH create_bvh() {
        BVH bvh;
        bvh.num_nodes = static_cast<int>(nodes.size());
        bvh.num_primitives = static_cast<int>(primitives.size());

        bvh.nodes = new BVHNode[nodes.size()];
        bvh.primitives = new HittableObject[primitives.size()];

        std::copy(nodes.begin(), nodes.end(), bvh.nodes);
        std::copy(primitives.begin(), primitives.end(), bvh.primitives);

        return bvh;
    }

    /** @brief Libere la mem alloc par create_bvh() */
    void free_bvh(BVH& bvh) {
        if (bvh.nodes) delete[] bvh.nodes;
        if (bvh.primitives) delete[] bvh.primitives;
        bvh.nodes = nullptr;
        bvh.primitives = nullptr;
    }

private:
    /** @brief Construction recursive du sous-arbre [start, end) */
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

/** @brief Main CPU - scene, BVH, rendu OpenMP, sauvegarde */
int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    args.use_cpu = true;

    RenderConfig config;
    config.width = args.width;
    config.height = args.height;
    config.samples_per_pixel = args.samples;
    config.max_depth = args.depth;

    int num_pixels = config.width * config.height;

    std::vector<HittableObject> h_objects(MAX_OBJECTS);
    std::vector<Material> h_materials(MAX_MATERIALS);
    int obj_count = 0;
    int mat_count = 0;

    Camera camera;
    HittableList world;

    create_default_scene(camera, world, h_objects.data(), h_materials.data(), obj_count, mat_count, config);

    if (!args.quiet) {
        std::cout << "CPU Ray Tracer (OpenMP)\n";
        std::cout << "Resolution: " << config.width << "x" << config.height << "\n";
        std::cout << "Samples: " << config.samples_per_pixel << "\n";
        std::cout << "Max Depth: " << config.max_depth << "\n";
        std::cout << "Objects: " << obj_count << "\n";
        std::cout << "Output: " << args.output_file << "\n\n";
    }

    if (!args.quiet) std::cout << "Construction du BVH..." << std::flush;
    CPUBVHBuilder bvh_builder;
    bvh_builder.build(h_objects.data(), obj_count);
    BVH bvh = bvh_builder.create_bvh();
    if (!args.quiet) std::cout << " OK (" << bvh_builder.nodes.size() << " noeuds)\n";

    std::vector<Color> h_frame_buffer(num_pixels);

    if (!args.quiet) std::cout << "Rendu CPU en cours..." << std::flush;
    auto start_time = std::chrono::high_resolution_clock::now();

    render_cpu(h_frame_buffer.data(), camera, bvh, config);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    if (!args.quiet) {
        std::cout << " OK\n";
        std::cout << "Temps: " << duration.count() / 1000.0f << " secondes\n";
        float mrays = (float)config.width * config.height * config.samples_per_pixel / 1000000.0f;
        std::cout << "Performance: " << mrays / (duration.count() / 1000.0f) << " MRays/s\n";
    }

    if (!args.quiet) std::cout << "Sauvegarde..." << std::flush;
    save_image(args.output_file, h_frame_buffer.data(), config.width, config.height);
    if (!args.quiet) std::cout << " OK\n";

    bvh_builder.free_bvh(bvh);

    return 0;
}
