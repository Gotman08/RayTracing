/**
 * CUDA Ray Tracer - Main Entry Point
 * Optimized for Romeo2025 HPC (GH200/H100)
 */

#include <iostream>
#include <chrono>
#include <ctime>
#include <vector>

// External libraries - define implementations only once
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Core includes
#include "raytracer/core/cuda_utils.cuh"
#include "raytracer/core/cuda_buffer.cuh"
#include "raytracer/acceleration/bvh_builder.cuh"
#include "raytracer/rendering/renderer.cuh"

// Modular components
#include "cli/args.cuh"
#include "io/scene_loader.cuh"
#include "io/image_writer.cuh"
#include "scene/default_scene.cuh"

using namespace rt;

// =============================================================================
// Scene Limits
// =============================================================================

static constexpr int MAX_OBJECTS = 1000;
static constexpr int MAX_MATERIALS = 1000;

// =============================================================================
// Material Pointer Helpers
// =============================================================================

inline Material* get_material_ptr(const HittableObject& obj) {
    switch (obj.type) {
        case HittableType::SPHERE:        return obj.data.sphere.mat;
        case HittableType::MOVING_SPHERE: return obj.data.moving_sphere.mat;
        case HittableType::QUAD:          return obj.data.quad.mat;
        case HittableType::BOX:           return obj.data.box.mat;
        case HittableType::TRIANGLE:      return obj.data.triangle.mat;
        case HittableType::PLANE:         return obj.data.plane.mat;
        default:                          return nullptr;
    }
}

inline void set_material_ptr(HittableObject& obj, Material* mat) {
    switch (obj.type) {
        case HittableType::SPHERE:
            obj.data.sphere.mat = mat;
            break;
        case HittableType::MOVING_SPHERE:
            obj.data.moving_sphere.mat = mat;
            break;
        case HittableType::QUAD:
            obj.data.quad.mat = mat;
            break;
        case HittableType::BOX:
            obj.data.box.mat = mat;
            for (int s = 0; s < 6; s++) {
                obj.data.box.sides[s].mat = mat;
            }
            break;
        case HittableType::TRIANGLE:
            obj.data.triangle.mat = mat;
            break;
        case HittableType::PLANE:
            obj.data.plane.mat = mat;
            break;
        default:
            break;
    }
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    if (args.show_info) {
        print_device_info();
        return 0;
    }

    // Configuration
    RenderConfig config;
    config.width = args.width;
    config.height = args.height;
    config.samples_per_pixel = args.samples;
    config.max_depth = args.depth;
    config.exposure = args.exposure;
    config.tone_map = ToneMapMode::ACES;

    int num_pixels = config.width * config.height;

    // Allocate host memory using std::vector (RAII)
    std::vector<HittableObject> h_objects(MAX_OBJECTS);
    std::vector<Material> h_materials(MAX_MATERIALS);
    int obj_count = 0;
    int mat_count = 0;

    Camera camera;
    HittableList world;

    // Load or create scene
    if (!args.scene_file.empty()) {
        if (!load_scene_from_json(args.scene_file, camera, world, h_objects.data(), h_materials.data(),
                                   obj_count, mat_count, config, MAX_OBJECTS, MAX_MATERIALS)) {
            std::cerr << "Failed to load scene, using default scene instead" << std::endl;
        }
    }

    if (obj_count == 0) {
        create_default_scene(camera, world, h_objects.data(), h_materials.data(), obj_count, mat_count, config);
    }

    // Validate scene limits
    if (obj_count > MAX_OBJECTS || mat_count > MAX_MATERIALS) {
        std::cerr << "Error: Scene exceeds limits (objects: " << obj_count << "/" << MAX_OBJECTS
                  << ", materials: " << mat_count << "/" << MAX_MATERIALS << ")" << std::endl;
        return 1;
    }

    if (!args.quiet) {
        std::cout << "CUDA Ray Tracer v1.0\n";
        std::cout << "Resolution: " << config.width << "x" << config.height << "\n";
        std::cout << "Samples: " << config.samples_per_pixel << "\n";
        std::cout << "Max Depth: " << config.max_depth << "\n";
        std::cout << "Objects: " << obj_count << "\n";
        std::cout << "Output: " << args.output_file << "\n\n";
    }

    // Allocate device memory using CudaBuffer (RAII)
    CudaBuffer<Color> d_frame_buffer(num_pixels);
    CudaBuffer<curandState> d_rand_states(num_pixels);
    CudaBuffer<Material> d_materials(mat_count);

    // Copy materials to device
    d_materials.copyFrom(h_materials.data(), mat_count);

    // Update material pointers to device memory (must be done BEFORE building BVH)
    for (int i = 0; i < obj_count; i++) {
        Material* host_mat_ptr = get_material_ptr(h_objects[i]);
        if (!host_mat_ptr) continue;

        for (int m = 0; m < mat_count; m++) {
            if (host_mat_ptr == &h_materials[m]) {
                set_material_ptr(h_objects[i], d_materials.get() + m);
                break;
            }
        }
    }

    // Build BVH
    if (!args.quiet) std::cout << "Building BVH..." << std::flush;
    BVHBuilder bvh_builder;
    bvh_builder.build(h_objects.data(), obj_count);
    BVH bvh = bvh_builder.create_gpu_bvh();
    if (!args.quiet) std::cout << " Done (" << bvh_builder.nodes.size() << " nodes).\n";

    // Kernel configuration
    constexpr int BLOCK_DIM_X = 16;
    constexpr int BLOCK_DIM_Y = 16;
    dim3 blocks((config.width + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
                (config.height + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
    dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y);

    // Initialize random states
    if (!args.quiet) std::cout << "Initializing random states..." << std::flush;
    init_render_rand_states<<<blocks, threads>>>(d_rand_states.get(), config.width, config.height, time(NULL));
    CUDA_SYNC_CHECK();
    if (!args.quiet) std::cout << " Done.\n";

    // Render
    if (!args.quiet) std::cout << "Rendering with BVH acceleration..." << std::flush;
    auto start_time = std::chrono::high_resolution_clock::now();

    render_kernel_bvh<<<blocks, threads>>>(d_frame_buffer.get(), camera, bvh, config, d_rand_states.get());
    CUDA_SYNC_CHECK();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    if (!args.quiet) {
        std::cout << " Done.\n";
        std::cout << "Render time: " << duration.count() / 1000.0f << " seconds\n";
        float mrays = (float)config.width * config.height * config.samples_per_pixel / 1000000.0f;
        std::cout << "Performance: " << mrays / (duration.count() / 1000.0f) << " MRays/s\n";
    }

    // Copy result back and save
    std::vector<Color> h_frame_buffer(num_pixels);
    d_frame_buffer.copyTo(h_frame_buffer.data());

    if (!args.quiet) std::cout << "Saving image..." << std::flush;
    save_image(args.output_file, h_frame_buffer.data(), config.width, config.height);
    if (!args.quiet) std::cout << " Done.\n";

    // Cleanup BVH (CudaBuffers and vectors auto-cleanup via RAII)
    bvh_builder.free_gpu_bvh(bvh);

    return 0;
}
