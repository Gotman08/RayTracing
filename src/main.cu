#include <iostream>
#include <chrono>
#include <ctime>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "raytracer/core/cuda_utils.cuh"
#include "raytracer/core/timer.cuh"
#include "raytracer/acceleration/bvh_builder.cuh"
#include "raytracer/rendering/renderer.cuh"
#include "raytracer/rendering/cpu_renderer.cuh"

#include "cli/args.cuh"
#include "io/image_writer.cuh"
#include "scene/default_scene.cuh"

#ifdef ENABLE_INTERACTIVE
#include "raytracer/interactive/window.cuh"
#include "raytracer/interactive/gl_interop.cuh"
#include "raytracer/interactive/camera_controller.cuh"
#include "raytracer/interactive/input_handler.cuh"
#include "raytracer/interactive/accumulation_buffer.cuh"
#endif

using namespace rt;

static constexpr int MAX_OBJECTS = 1000;
static constexpr int MAX_MATERIALS = 1000;

#ifdef ENABLE_INTERACTIVE
int run_interactive_mode(
    const Args& args,
    Camera& initial_camera,
    BVH& bvh,
    RenderConfig& config,
    curandState* d_rand_states
) {
    // Initialize GLFW window
    Window window;
    if (!window.initialize(config.width, config.height, "CUDA Ray Tracer - Interactive")) {
        std::cerr << "Failed to initialize window\n";
        return 1;
    }

    // Initialize CUDA-GL interop
    GLInterop gl_interop;
    if (!gl_interop.initialize(config.width, config.height)) {
        std::cerr << "Failed to initialize GL interop\n";
        window.cleanup();
        return 1;
    }

    // Initialize accumulation buffer
    AccumulationBuffer accum_buffer;
    if (!accum_buffer.initialize(config.width, config.height)) {
        std::cerr << "Failed to initialize accumulation buffer\n";
        gl_interop.cleanup();
        window.cleanup();
        return 1;
    }

    // Initialize camera controller from initial camera
    CameraController camera_ctrl;
    camera_ctrl.initialize(
        initial_camera.center,
        initial_camera.center - initial_camera.w * 10.0f,
        20.0f,
        initial_camera.defocus_angle,
        10.0f
    );
    camera_ctrl.move_speed = args.move_speed;
    camera_ctrl.mouse_sensitivity = args.mouse_sensitivity;

    // Initialize input handler
    InputHandler input;
    input.setup_callbacks(window.handle);

    // Rendering setup
    constexpr int BLOCK_SIZE = 16;
    dim3 blocks((config.width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (config.height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    // Frame timing
    auto last_frame_time = std::chrono::high_resolution_clock::now();
    float fps = 0.0f;
    int frame_count = 0;
    auto fps_timer = std::chrono::high_resolution_clock::now();

    std::cout << "Interactive mode started\n";
    std::cout << "  WASD: Move | Mouse: Look | Space/Q: Up | Ctrl/E: Down\n";
    std::cout << "  Shift: Fast | R: Reset | P: Screenshot | Esc: Quit\n\n";

    // Main loop
    while (!window.should_close() && !input.state.should_close) {
        // Calculate delta time
        auto current_time = std::chrono::high_resolution_clock::now();
        float delta_time = std::chrono::duration<float>(current_time - last_frame_time).count();
        last_frame_time = current_time;

        // Poll input
        input.poll_events();

        // Update camera
        bool camera_moved = camera_ctrl.update(input.state, delta_time);

        // Reset accumulation if camera moved or R pressed
        if (camera_moved || input.state.reset_accumulation) {
            accum_buffer.reset();
        }

        // Determine if we should render more samples
        bool should_render = (accum_buffer.get_accumulated_samples() < args.max_accumulated_spp);

        if (should_render) {
            // Build camera for this frame
            Camera frame_camera = camera_ctrl.build_camera(config.width, config.height);

            // Render to accumulation buffer
            render_kernel_accumulate<<<blocks, threads>>>(
                accum_buffer.d_buffer,
                frame_camera,
                bvh,
                config,
                d_rand_states,
                args.interactive_spp
            );
            cudaDeviceSynchronize();

            accum_buffer.accumulated_samples += args.interactive_spp;
        }

        // Convert accumulation buffer to display format
        uchar4* d_display = gl_interop.map_for_cuda();

        convert_to_rgba8<<<blocks, threads>>>(
            d_display,
            accum_buffer.d_buffer,
            config.width, config.height,
            accum_buffer.get_accumulated_samples()
        );
        cudaDeviceSynchronize();

        gl_interop.unmap_from_cuda();

        // Display
        glClear(GL_COLOR_BUFFER_BIT);
        gl_interop.display();
        window.swap_buffers();

        // Handle screenshot request
        if (input.state.screenshot_requested) {
            std::vector<Color> h_buffer(config.width * config.height);
            cudaMemcpy(h_buffer.data(), accum_buffer.d_buffer,
                       config.width * config.height * sizeof(Color),
                       cudaMemcpyDeviceToHost);

            // Normalize and tone map
            int samples = accum_buffer.get_accumulated_samples();
            for (auto& c : h_buffer) {
                if (samples > 0) c = c / static_cast<float>(samples);
                c = Color(c.x / (1.0f + c.x), c.y / (1.0f + c.y), c.z / (1.0f + c.z));
                c = Color(powf(fmaxf(0.0f, c.x), 1.0f/2.2f),
                          powf(fmaxf(0.0f, c.y), 1.0f/2.2f),
                          powf(fmaxf(0.0f, c.z), 1.0f/2.2f));
            }

            save_image("screenshot.png", h_buffer.data(), config.width, config.height);
            std::cout << "Screenshot saved: screenshot.png (" << samples << " SPP)\n";
        }

        // Update FPS counter
        frame_count++;
        auto fps_elapsed = std::chrono::duration<float>(current_time - fps_timer).count();
        if (fps_elapsed >= 1.0f) {
            fps = frame_count / fps_elapsed;
            frame_count = 0;
            fps_timer = current_time;

            char title[256];
            snprintf(title, sizeof(title),
                     "CUDA Ray Tracer - %.1f FPS - %d/%d SPP",
                     fps, accum_buffer.get_accumulated_samples(), args.max_accumulated_spp);
            window.set_title(title);
        }

        input.state.clear_deltas();
    }

    // Cleanup
    accum_buffer.cleanup();
    gl_interop.cleanup();
    window.cleanup();

    return 0;
}
#endif

inline Material* get_material_ptr(const HittableObject& obj) {
    switch (obj.type) {
        case HittableType::SPHERE: return obj.data.sphere.mat;
        case HittableType::PLANE:  return obj.data.plane.mat;
        default:                   return nullptr;
    }
}

inline void set_material_ptr(HittableObject& obj, Material* mat) {
    switch (obj.type) {
        case HittableType::SPHERE:
            obj.data.sphere.mat = mat;
            break;
        case HittableType::PLANE:
            obj.data.plane.mat = mat;
            break;
        default:
            break;
    }
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    if (args.show_info) {
        print_device_info();
        return 0;
    }

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
        std::cout << (args.use_cpu ? "CPU" : "CUDA") << " Ray Tracer\n";
        std::cout << "Resolution: " << config.width << "x" << config.height << "\n";
        std::cout << "Samples: " << config.samples_per_pixel << "\n";
        std::cout << "Max Depth: " << config.max_depth << "\n";
        std::cout << "Objects: " << obj_count << "\n";
        std::cout << "Output: " << args.output_file << "\n\n";
    }

    // CPU Rendering Path
    if (args.use_cpu) {
        Timer timer;

        timer.start("BVH Construction");
        if (!args.quiet) std::cout << "Construction du BVH..." << std::flush;
        BVHBuilder bvh_builder;
        bvh_builder.build(h_objects.data(), obj_count);
        BVH bvh = bvh_builder.create_cpu_bvh();
        timer.stop();
        if (!args.quiet) std::cout << " OK (" << bvh_builder.nodes.size() << " noeuds)\n";

        timer.start("Buffer Allocation");
        std::vector<Color> h_frame_buffer(num_pixels);
        timer.stop();

        if (!args.quiet) std::cout << "Rendu CPU en cours..." << std::flush;
        timer.start("Render (CPU/OpenMP)");
        auto start_time = std::chrono::high_resolution_clock::now();

        render_cpu(h_frame_buffer.data(), camera, bvh, config);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        timer.stop();

        if (!args.quiet) {
            std::cout << " OK\n";
            std::cout << "Temps: " << duration.count() / 1000.0f << " secondes\n";
            float mrays = (float)config.width * config.height * config.samples_per_pixel / 1000000.0f;
            std::cout << "Performance: " << mrays / (duration.count() / 1000.0f) << " MRays/s\n";
        }

        timer.start("Save Image");
        if (!args.quiet) std::cout << "Sauvegarde..." << std::flush;
        save_image(args.output_file, h_frame_buffer.data(), config.width, config.height);
        timer.stop();
        if (!args.quiet) std::cout << " OK\n";

        if (args.profile) {
            timer.print_report("CPU Profiling");
        }

        bvh_builder.free_cpu_bvh(bvh);
        return 0;
    }

    // GPU Rendering Path
    Timer timer;
    CudaTimer cuda_timer;

    timer.start("GPU Memory Allocation");
    Color* d_frame_buffer;
    curandState* d_rand_states;
    Material* d_materials;

    cudaMalloc(&d_frame_buffer, num_pixels * sizeof(Color));
    cudaMalloc(&d_rand_states, num_pixels * sizeof(curandState));
    cudaMalloc(&d_materials, mat_count * sizeof(Material));
    timer.stop();

    timer.start("Materials Upload (H2D)");
    cudaMemcpy(d_materials, h_materials.data(), mat_count * sizeof(Material), cudaMemcpyHostToDevice);
    timer.stop();

    timer.start("Pointer Fixup");
    for (int i = 0; i < obj_count; i++) {
        Material* host_mat_ptr = get_material_ptr(h_objects[i]);
        if (!host_mat_ptr) continue;

        for (int m = 0; m < mat_count; m++) {
            if (host_mat_ptr == &h_materials[m]) {
                set_material_ptr(h_objects[i], d_materials + m);
                break;
            }
        }
    }
    timer.stop();

    if (!args.quiet) std::cout << "Construction du BVH..." << std::flush;
    timer.start("BVH Build (CPU)");
    BVHBuilder bvh_builder;
    bvh_builder.build(h_objects.data(), obj_count);
    timer.stop();

    timer.start("BVH Upload (H2D)");
    BVH bvh = bvh_builder.create_gpu_bvh();
    timer.stop();
    if (!args.quiet) std::cout << " OK (" << bvh_builder.nodes.size() << " noeuds)\n";

    constexpr int BLOCK_SIZE = 16;
    dim3 blocks((config.width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (config.height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    if (!args.quiet) std::cout << "Initialisation..." << std::flush;
    timer.start("Init Random States");
    cuda_timer.start();
    // Use fast hash-based initialization
    init_rand_states_fast<<<blocks, threads>>>(d_rand_states, config.width, config.height, static_cast<unsigned int>(time(NULL)));
    CUDA_SYNC_CHECK();
    cuda_timer.stop();
    timer.stop();
    if (!args.quiet) std::cout << " OK\n";

#ifdef ENABLE_INTERACTIVE
    if (args.interactive) {
        int result = run_interactive_mode(args, camera, bvh, config, d_rand_states);

        cudaFree(d_frame_buffer);
        cudaFree(d_rand_states);
        cudaFree(d_materials);
        bvh_builder.free_gpu_bvh(bvh);

        return result;
    }
#else
    if (args.interactive) {
        std::cerr << "Interactive mode not enabled. Rebuild with -DENABLE_INTERACTIVE=ON\n";
        cudaFree(d_frame_buffer);
        cudaFree(d_rand_states);
        cudaFree(d_materials);
        bvh_builder.free_gpu_bvh(bvh);
        return 1;
    }
#endif

    if (!args.quiet) std::cout << "Rendu en cours..." << std::flush;
    timer.start("Render Kernel (GPU)");
    cuda_timer.start();
    auto start_time = std::chrono::high_resolution_clock::now();

    render_kernel<<<blocks, threads>>>(d_frame_buffer, camera, bvh, config, d_rand_states);
    CUDA_SYNC_CHECK();

    cuda_timer.stop();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    timer.stop();

    float kernel_time_ms = cuda_timer.elapsed_ms();

    if (!args.quiet) {
        std::cout << " OK\n";
        std::cout << "Temps: " << duration.count() / 1000.0f << " secondes\n";
        float mrays = (float)config.width * config.height * config.samples_per_pixel / 1000000.0f;
        std::cout << "Performance: " << mrays / (duration.count() / 1000.0f) << " MRays/s\n";
    }

    timer.start("Framebuffer Download (D2H)");
    std::vector<Color> h_frame_buffer(num_pixels);
    cudaMemcpy(h_frame_buffer.data(), d_frame_buffer, num_pixels * sizeof(Color), cudaMemcpyDeviceToHost);
    timer.stop();

    timer.start("Save Image");
    if (!args.quiet) std::cout << "Sauvegarde..." << std::flush;
    save_image(args.output_file, h_frame_buffer.data(), config.width, config.height);
    timer.stop();
    if (!args.quiet) std::cout << " OK\n";

    if (args.profile) {
        timer.print_report("GPU Profiling");
        std::cout << "  Kernel GPU time (CUDA events): " << kernel_time_ms << " ms\n\n";
    }

    cudaFree(d_frame_buffer);
    cudaFree(d_rand_states);
    cudaFree(d_materials);
    bvh_builder.free_gpu_bvh(bvh);

    return 0;
}
