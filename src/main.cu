/**
 * CUDA Ray Tracer - Main Entry Point
 * Optimized for Romeo2025 HPC (GH200/H100)
 */

#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <cstring>

// External libraries
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "json.hpp"

// Ray tracer includes
#include "raytracer/core/cuda_utils.cuh"
#include "raytracer/core/vec3.cuh"
#include "raytracer/core/ray.cuh"
#include "raytracer/geometry/hittable_list.cuh"
#include "raytracer/materials/lambertian.cuh"
#include "raytracer/materials/metal.cuh"
#include "raytracer/materials/dielectric.cuh"
#include "raytracer/materials/emissive.cuh"
#include "raytracer/camera/camera.cuh"
#include "raytracer/rendering/renderer.cuh"
#include "raytracer/rendering/integrator.cuh"

using json = nlohmann::json;
using namespace rt;

// =============================================================================
// Command Line Arguments
// =============================================================================

struct Args {
    int width = 800;
    int height = 600;
    int samples = 100;
    int depth = 50;
    std::string scene_file = "";
    std::string output_file = "output.png";
    std::string hdri_file = "";
    bool show_info = false;
    bool quiet = false;
    float exposure = 0.0f;
};

void print_usage(const char* program) {
    std::cout << "CUDA Ray Tracer v1.0 - Optimized for Romeo2025 (GH200/H100)\n\n";
    std::cout << "Usage: " << program << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -w, --width <int>      Image width (default: 800)\n";
    std::cout << "  -h, --height <int>     Image height (default: 600)\n";
    std::cout << "  -s, --samples <int>    Samples per pixel (default: 100)\n";
    std::cout << "  -d, --depth <int>      Max ray bounces (default: 50)\n";
    std::cout << "  -i, --scene <file>     Scene file (JSON format)\n";
    std::cout << "  -o, --output <file>    Output file (default: output.png)\n";
    std::cout << "  --hdri <file>          HDR environment map\n";
    std::cout << "  --exposure <float>     Exposure adjustment (default: 0.0)\n";
    std::cout << "  --info                 Display GPU information\n";
    std::cout << "  --quiet                Suppress progress output\n";
    std::cout << "  --help                 Show this help\n";
}

Args parse_args(int argc, char** argv) {
    Args args;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        } else if (arg == "--info") {
            args.show_info = true;
        } else if (arg == "--quiet") {
            args.quiet = true;
        } else if ((arg == "-w" || arg == "--width") && i + 1 < argc) {
            args.width = std::stoi(argv[++i]);
        } else if ((arg == "-h" || arg == "--height") && i + 1 < argc) {
            args.height = std::stoi(argv[++i]);
        } else if ((arg == "-s" || arg == "--samples") && i + 1 < argc) {
            args.samples = std::stoi(argv[++i]);
        } else if ((arg == "-d" || arg == "--depth") && i + 1 < argc) {
            args.depth = std::stoi(argv[++i]);
        } else if ((arg == "-i" || arg == "--scene") && i + 1 < argc) {
            args.scene_file = argv[++i];
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            args.output_file = argv[++i];
        } else if (arg == "--hdri" && i + 1 < argc) {
            args.hdri_file = argv[++i];
        } else if (arg == "--exposure" && i + 1 < argc) {
            args.exposure = std::stof(argv[++i]);
        }
    }

    return args;
}

// =============================================================================
// Scene Loading
// =============================================================================

Material* create_material_from_json(const json& mat_json, Material* materials, int& mat_count) {
    std::string type = mat_json.value("type", "lambertian");
    Material* mat = &materials[mat_count++];

    if (type == "lambertian") {
        auto c = mat_json.value("color", std::vector<float>{0.5f, 0.5f, 0.5f});
        *mat = create_lambertian(Color(c[0], c[1], c[2]));
    } else if (type == "metal") {
        auto c = mat_json.value("color", std::vector<float>{0.8f, 0.8f, 0.8f});
        float fuzz = mat_json.value("fuzz", 0.0f);
        *mat = create_metal(Color(c[0], c[1], c[2]), fuzz);
    } else if (type == "dielectric" || type == "glass") {
        float ior = mat_json.value("ior", 1.5f);
        *mat = create_dielectric(ior);
    } else if (type == "emissive" || type == "light") {
        auto c = mat_json.value("color", std::vector<float>{1.0f, 1.0f, 1.0f});
        float strength = mat_json.value("strength", 1.0f);
        *mat = create_emissive(Color(c[0], c[1], c[2]), strength);
    } else if (type == "checker") {
        auto c1 = mat_json.value("color1", std::vector<float>{0.2f, 0.3f, 0.1f});
        auto c2 = mat_json.value("color2", std::vector<float>{0.9f, 0.9f, 0.9f});
        float scale = mat_json.value("scale", 10.0f);
        *mat = create_lambertian_checker(scale, Color(c1[0], c1[1], c1[2]), Color(c2[0], c2[1], c2[2]));
    } else {
        *mat = create_lambertian(Color(0.5f, 0.5f, 0.5f));
    }

    return mat;
}

void load_scene_from_json(
    const std::string& filename,
    Camera& camera,
    HittableList& world,
    HittableObject* objects,
    Material* materials,
    int& obj_count,
    int& mat_count,
    RenderConfig& config
) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open scene file: " << filename << std::endl;
        return;
    }

    json scene = json::parse(file);

    // Camera
    if (scene.contains("camera")) {
        auto& cam = scene["camera"];
        auto lookfrom = cam.value("lookfrom", std::vector<float>{13, 2, 3});
        auto lookat = cam.value("lookat", std::vector<float>{0, 0, 0});
        auto vup = cam.value("vup", std::vector<float>{0, 1, 0});
        float vfov = cam.value("fov", 20.0f);
        float aperture = cam.value("aperture", 0.0f);
        float focus_dist = cam.value("focus_dist", 10.0f);

        camera.initialize(
            config.width, config.height,
            Point3(lookfrom[0], lookfrom[1], lookfrom[2]),
            Point3(lookat[0], lookat[1], lookat[2]),
            Vec3(vup[0], vup[1], vup[2]),
            vfov, aperture, focus_dist
        );
    }

    // Background
    if (scene.contains("background")) {
        auto bg = scene["background"];
        if (bg.is_array()) {
            config.background = Color(bg[0], bg[1], bg[2]);
            config.use_sky = false;
        } else if (bg.is_string() && bg == "sky") {
            config.use_sky = true;
        }
    }

    // Objects
    if (scene.contains("objects")) {
        for (auto& obj : scene["objects"]) {
            std::string type = obj.value("type", "sphere");

            Material* mat = nullptr;
            if (obj.contains("material")) {
                mat = create_material_from_json(obj["material"], materials, mat_count);
            } else {
                mat = &materials[mat_count++];
                *mat = create_lambertian(Color(0.5f, 0.5f, 0.5f));
            }

            if (type == "sphere") {
                auto center = obj.value("center", std::vector<float>{0, 0, 0});
                float radius = obj.value("radius", 1.0f);

                objects[obj_count].type = HittableType::SPHERE;
                objects[obj_count].data.sphere = Sphere(
                    Point3(center[0], center[1], center[2]),
                    radius, mat
                );
                objects[obj_count].bbox = objects[obj_count].data.sphere.bounding_box();
                obj_count++;

            } else if (type == "quad") {
                auto Q = obj.value("Q", std::vector<float>{0, 0, 0});
                auto u = obj.value("u", std::vector<float>{1, 0, 0});
                auto v = obj.value("v", std::vector<float>{0, 1, 0});

                objects[obj_count].type = HittableType::QUAD;
                objects[obj_count].data.quad = Quad(
                    Point3(Q[0], Q[1], Q[2]),
                    Vec3(u[0], u[1], u[2]),
                    Vec3(v[0], v[1], v[2]),
                    mat
                );
                objects[obj_count].bbox = objects[obj_count].data.quad.bounding_box();
                obj_count++;

            } else if (type == "box") {
                auto a = obj.value("min", std::vector<float>{0, 0, 0});
                auto b = obj.value("max", std::vector<float>{1, 1, 1});

                objects[obj_count].type = HittableType::BOX;
                objects[obj_count].data.box = Box(
                    Point3(a[0], a[1], a[2]),
                    Point3(b[0], b[1], b[2]),
                    mat
                );
                objects[obj_count].bbox = objects[obj_count].data.box.bounding_box();
                obj_count++;
            }
        }
    }
}

// =============================================================================
// Default Scenes
// =============================================================================

void create_default_scene(
    Camera& camera,
    HittableList& world,
    HittableObject* objects,
    Material* materials,
    int& obj_count,
    int& mat_count,
    RenderConfig& config
) {
    // Camera
    camera.initialize(
        config.width, config.height,
        Point3(13, 2, 3),
        Point3(0, 0, 0),
        Vec3(0, 1, 0),
        20.0f, 0.1f, 10.0f
    );

    // Ground (checker)
    materials[mat_count] = create_lambertian_checker(10.0f, Color(0.2f, 0.3f, 0.1f), Color(0.9f, 0.9f, 0.9f));
    objects[obj_count].type = HittableType::SPHERE;
    objects[obj_count].data.sphere = Sphere(Point3(0, -1000, 0), 1000, &materials[mat_count]);
    objects[obj_count].bbox = objects[obj_count].data.sphere.bounding_box();
    mat_count++; obj_count++;

    // Glass sphere
    materials[mat_count] = create_dielectric(1.5f);
    objects[obj_count].type = HittableType::SPHERE;
    objects[obj_count].data.sphere = Sphere(Point3(0, 1, 0), 1.0f, &materials[mat_count]);
    objects[obj_count].bbox = objects[obj_count].data.sphere.bounding_box();
    mat_count++; obj_count++;

    // Diffuse sphere
    materials[mat_count] = create_lambertian(Color(0.4f, 0.2f, 0.1f));
    objects[obj_count].type = HittableType::SPHERE;
    objects[obj_count].data.sphere = Sphere(Point3(-4, 1, 0), 1.0f, &materials[mat_count]);
    objects[obj_count].bbox = objects[obj_count].data.sphere.bounding_box();
    mat_count++; obj_count++;

    // Metal sphere
    materials[mat_count] = create_metal(Color(0.7f, 0.6f, 0.5f), 0.0f);
    objects[obj_count].type = HittableType::SPHERE;
    objects[obj_count].data.sphere = Sphere(Point3(4, 1, 0), 1.0f, &materials[mat_count]);
    objects[obj_count].bbox = objects[obj_count].data.sphere.bounding_box();
    mat_count++; obj_count++;

    // Small random spheres
    for (int a = -5; a < 5; a++) {
        for (int b = -5; b < 5; b++) {
            float choose_mat = (float)rand() / RAND_MAX;
            Point3 center(a + 0.9f * (float)rand() / RAND_MAX, 0.2f, b + 0.9f * (float)rand() / RAND_MAX);

            if ((center - Point3(4, 0.2f, 0)).length() > 0.9f) {
                if (choose_mat < 0.6f) {
                    // Diffuse
                    Color albedo(
                        (float)rand() / RAND_MAX * (float)rand() / RAND_MAX,
                        (float)rand() / RAND_MAX * (float)rand() / RAND_MAX,
                        (float)rand() / RAND_MAX * (float)rand() / RAND_MAX
                    );
                    materials[mat_count] = create_lambertian(albedo);
                } else if (choose_mat < 0.85f) {
                    // Metal
                    Color albedo(
                        0.5f + 0.5f * (float)rand() / RAND_MAX,
                        0.5f + 0.5f * (float)rand() / RAND_MAX,
                        0.5f + 0.5f * (float)rand() / RAND_MAX
                    );
                    float fuzz = 0.5f * (float)rand() / RAND_MAX;
                    materials[mat_count] = create_metal(albedo, fuzz);
                } else {
                    // Glass
                    materials[mat_count] = create_dielectric(1.5f);
                }

                objects[obj_count].type = HittableType::SPHERE;
                objects[obj_count].data.sphere = Sphere(center, 0.2f, &materials[mat_count]);
                objects[obj_count].bbox = objects[obj_count].data.sphere.bounding_box();
                mat_count++;
                obj_count++;
            }
        }
    }

    config.use_sky = true;
}

// =============================================================================
// Image Output
// =============================================================================

void save_image(const std::string& filename, Color* buffer, int width, int height) {
    std::vector<unsigned char> pixels(width * height * 3);

    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            int idx = j * width + i;
            pixels[idx * 3 + 0] = static_cast<unsigned char>(255.999f * buffer[idx].x);
            pixels[idx * 3 + 1] = static_cast<unsigned char>(255.999f * buffer[idx].y);
            pixels[idx * 3 + 2] = static_cast<unsigned char>(255.999f * buffer[idx].z);
        }
    }

    std::string ext = filename.substr(filename.find_last_of('.') + 1);

    if (ext == "png") {
        stbi_write_png(filename.c_str(), width, height, 3, pixels.data(), width * 3);
    } else if (ext == "jpg" || ext == "jpeg") {
        stbi_write_jpg(filename.c_str(), width, height, 3, pixels.data(), 95);
    } else {
        // Default to PPM
        std::ofstream file(filename);
        file << "P3\n" << width << " " << height << "\n255\n";
        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) {
                int idx = j * width + i;
                file << static_cast<int>(255.999f * buffer[idx].x) << " "
                     << static_cast<int>(255.999f * buffer[idx].y) << " "
                     << static_cast<int>(255.999f * buffer[idx].z) << "\n";
            }
        }
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

    // Allocate host memory for scene
    const int MAX_OBJECTS = 1000;
    const int MAX_MATERIALS = 1000;

    HittableObject* h_objects = new HittableObject[MAX_OBJECTS];
    Material* h_materials = new Material[MAX_MATERIALS];
    int obj_count = 0;
    int mat_count = 0;

    // Create camera
    Camera camera;

    // Create world
    HittableList world;

    // Load or create scene
    if (!args.scene_file.empty()) {
        load_scene_from_json(args.scene_file, camera, world, h_objects, h_materials, obj_count, mat_count, config);
    }

    if (obj_count == 0) {
        create_default_scene(camera, world, h_objects, h_materials, obj_count, mat_count, config);
    }

    if (!args.quiet) {
        std::cout << "CUDA Ray Tracer v1.0\n";
        std::cout << "Resolution: " << config.width << "x" << config.height << "\n";
        std::cout << "Samples: " << config.samples_per_pixel << "\n";
        std::cout << "Max Depth: " << config.max_depth << "\n";
        std::cout << "Objects: " << obj_count << "\n";
        std::cout << "Output: " << args.output_file << "\n\n";
    }

    // Allocate device memory
    Color* d_frame_buffer;
    curandState* d_rand_states;
    HittableObject* d_objects;
    Material* d_materials;

    CUDA_CHECK(cudaMalloc(&d_frame_buffer, num_pixels * sizeof(Color)));
    CUDA_CHECK(cudaMalloc(&d_rand_states, num_pixels * sizeof(curandState)));
    CUDA_CHECK(cudaMalloc(&d_objects, obj_count * sizeof(HittableObject)));
    CUDA_CHECK(cudaMalloc(&d_materials, mat_count * sizeof(Material)));

    // Copy scene to device
    CUDA_CHECK(cudaMemcpy(d_objects, h_objects, obj_count * sizeof(HittableObject), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_materials, h_materials, mat_count * sizeof(Material), cudaMemcpyHostToDevice));

    // Update material pointers on device
    for (int i = 0; i < obj_count; i++) {
        // Find material index and update pointer
        for (int m = 0; m < mat_count; m++) {
            if (h_objects[i].data.sphere.mat == &h_materials[m]) {
                // Update to device pointer offset
                Material* device_mat_ptr = d_materials + m;
                size_t offset = offsetof(HittableObject, data);

                // This is simplified - in production, handle each type
                if (h_objects[i].type == HittableType::SPHERE) {
                    Sphere temp = h_objects[i].data.sphere;
                    temp.mat = device_mat_ptr;
                    CUDA_CHECK(cudaMemcpy(&d_objects[i].data.sphere, &temp, sizeof(Sphere), cudaMemcpyHostToDevice));
                }
                break;
            }
        }
    }

    // Setup world on device
    world.objects = d_objects;
    world.count = obj_count;
    world.capacity = obj_count;

    // Compute bounding box
    if (obj_count > 0) {
        world.bbox = h_objects[0].bbox;
        for (int i = 1; i < obj_count; i++) {
            world.bbox = AABB(world.bbox, h_objects[i].bbox);
        }
    }

    // Kernel configuration
    int tx = 8, ty = 8;
    dim3 blocks((config.width + tx - 1) / tx, (config.height + ty - 1) / ty);
    dim3 threads(tx, ty);

    // Initialize random states
    if (!args.quiet) std::cout << "Initializing random states..." << std::flush;
    init_render_rand_states<<<blocks, threads>>>(d_rand_states, config.width, config.height, time(NULL));
    CUDA_SYNC_CHECK();
    if (!args.quiet) std::cout << " Done.\n";

    // Render
    if (!args.quiet) std::cout << "Rendering..." << std::flush;
    auto start_time = std::chrono::high_resolution_clock::now();

    render_kernel<<<blocks, threads>>>(d_frame_buffer, camera, world, config, d_rand_states);
    CUDA_SYNC_CHECK();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    if (!args.quiet) {
        std::cout << " Done.\n";
        std::cout << "Render time: " << duration.count() / 1000.0f << " seconds\n";

        float mrays = (float)config.width * config.height * config.samples_per_pixel / 1000000.0f;
        std::cout << "Performance: " << mrays / (duration.count() / 1000.0f) << " MRays/s\n";
    }

    // Copy result back
    Color* h_frame_buffer = new Color[num_pixels];
    CUDA_CHECK(cudaMemcpy(h_frame_buffer, d_frame_buffer, num_pixels * sizeof(Color), cudaMemcpyDeviceToHost));

    // Save image
    if (!args.quiet) std::cout << "Saving image..." << std::flush;
    save_image(args.output_file, h_frame_buffer, config.width, config.height);
    if (!args.quiet) std::cout << " Done.\n";

    // Cleanup
    delete[] h_frame_buffer;
    delete[] h_objects;
    delete[] h_materials;

    CUDA_CHECK(cudaFree(d_frame_buffer));
    CUDA_CHECK(cudaFree(d_rand_states));
    CUDA_CHECK(cudaFree(d_objects));
    CUDA_CHECK(cudaFree(d_materials));

    return 0;
}
