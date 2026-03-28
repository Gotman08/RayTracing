/** @file main.cu
 * @brief Point d'entree du ray tracer CUDA (pipeline complet GPU/CPU) */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <vector>

// Thrust : librairie CUDA haut niveau pour les operations paralleles
// Fournie avec le CUDA Toolkit, pas de dependance externe
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>

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

using namespace rt;

/** @brief Foncteur luminance pour thrust::transform_reduce */
struct LuminanceFunctor {
    __host__ __device__ float operator()(const Color& c) const {
        return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
    }
};

/** @brief Recupere le ptr materiau d'un HittableObject (sphere/plan) */
inline Material* get_material_ptr(const HittableObject& obj) {
    switch (obj.type) {
        case HittableType::SPHERE: return obj.data.sphere.mat;
        case HittableType::PLANE:  return obj.data.plane.mat;
        default:                   return nullptr;
    }
}

/** @brief Remplace le ptr materiau (fixup host->device) */
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

/** @brief Main - orchestre scene, BVH, alloc GPU, rendu, sauvegarde */
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

    // allocation dynamique basee sur le nombre exact d'objets de la scene
    // count_scene_objects() retourne SCENE_TOTAL + 10 = 90 (voir default_scene.cuh)
    int estimated_objects = count_scene_objects();
    std::vector<HittableObject> h_objects(estimated_objects);
    std::vector<Material> h_materials(estimated_objects);
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

    Timer timer;
    CudaTimer cuda_timer;

    // -----------------------------------------------
    // Creation des streams CUDA pour le recouvrement d'operations
    // stream_compute : pour les kernels de calcul (init RNG, rendu)
    // stream_transfer : pour les transferts memoire (H2D, D2H)
    // -----------------------------------------------
    cudaStream_t stream_compute, stream_transfer;
    cudaStreamCreate(&stream_compute);
    cudaStreamCreate(&stream_transfer);

    timer.start("GPU Memory Allocation");
    Color* d_frame_buffer;
    curandState* d_rand_states;
    Material* d_materials;
    float* d_total_luminance;  // pour la reduction (un seul float)

    cudaMalloc(&d_frame_buffer, num_pixels * sizeof(Color));
    cudaMalloc(&d_rand_states, num_pixels * sizeof(curandState));
    cudaMalloc(&d_materials, mat_count * sizeof(Material));
    cudaMalloc(&d_total_luminance, sizeof(float));
    timer.stop();

    // copier Camera et RenderConfig en memoire constante GPU
    // → broadcast cache, lecture ~5 cycles vs ~500 en globale
    timer.start("Constant Memory Upload");
    cudaMemcpyToSymbol(d_const_camera, &camera, sizeof(Camera));
    cudaMemcpyToSymbol(d_const_config, &config, sizeof(RenderConfig));
    timer.stop();

    // -----------------------------------------------
    // Pointer fixup : les pointeurs materiaux host → device
    // Doit etre fait AVANT l'upload des objets au BVH
    // -----------------------------------------------
    timer.start("Materials Upload (H2D)");
    cudaMemcpy(d_materials, h_materials.data(),
               mat_count * sizeof(Material), cudaMemcpyHostToDevice);
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

    // -----------------------------------------------
    // Overlap avec streams : init RNG (compute) pendant que le BVH upload
    // vient de finir. Le stream_compute lance l'init en async.
    // -----------------------------------------------
    if (!args.quiet) std::cout << "Initialisation..." << std::flush;
    timer.start("Init Random States");
    cuda_timer.start();
    init_rand_states_fast<<<blocks, threads, 0, stream_compute>>>(
        d_rand_states, config.width, config.height,
        static_cast<unsigned int>(time(NULL)));
    cudaStreamSynchronize(stream_compute);
    cuda_timer.stop();
    CUDA_SYNC_CHECK();
    timer.stop();
    if (!args.quiet) std::cout << " OK\n";

    // -----------------------------------------------
    // Mode benchmark multi-resolution
    // Lance le rendu sur 4 resolutions differentes avec 50 SPP chacune,
    // mesure le temps avec CUDA events et affiche un tableau comparatif.
    // -----------------------------------------------
    if (args.benchmark) {
        struct Resolution { int w, h; const char* name; };
        const Resolution resolutions[] = {
            { 640,  360,  "640x360  (230K)" },
            { 1280, 720,  "1280x720 (922K)" },
            { 1920, 1080, "1920x1080 (2.1M)" },
            { 2560, 1440, "2560x1440 (3.7M)" },
        };
        const int BENCH_SPP = 50;

        std::cout << "\n=== Benchmark Multi-Resolution (" << BENCH_SPP << " SPP) ===\n";
        std::cout << std::left
                  << std::setw(20) << "Resolution"
                  << std::setw(12) << "Pixels"
                  << std::setw(14) << "Time (ms)"
                  << std::setw(12) << "MRays/s" << "\n";
        std::cout << std::string(58, '-') << "\n";

        for (const auto& res : resolutions) {
            int bench_pixels = res.w * res.h;

            // allouer les buffers pour cette resolution
            Color* d_bench_fb;
            curandState* d_bench_rng;
            cudaMalloc(&d_bench_fb, bench_pixels * sizeof(Color));
            cudaMalloc(&d_bench_rng, bench_pixels * sizeof(curandState));

            // mettre a jour la config en memoire constante
            RenderConfig bench_config = config;
            bench_config.width  = res.w;
            bench_config.height = res.h;
            bench_config.samples_per_pixel = BENCH_SPP;
            cudaMemcpyToSymbol(d_const_config, &bench_config, sizeof(RenderConfig));

            // init RNG pour cette resolution
            dim3 b_blocks((res.w + BLOCK_SIZE - 1) / BLOCK_SIZE,
                          (res.h + BLOCK_SIZE - 1) / BLOCK_SIZE);
            dim3 b_threads(BLOCK_SIZE, BLOCK_SIZE);
            init_rand_states_fast<<<b_blocks, b_threads>>>(
                d_bench_rng, res.w, res.h, 42u);
            cudaDeviceSynchronize();

            // timing precis avec CUDA events
            cudaEvent_t ev_start, ev_stop;
            cudaEventCreate(&ev_start);
            cudaEventCreate(&ev_stop);

            cudaEventRecord(ev_start);
            render_kernel<<<b_blocks, b_threads>>>(d_bench_fb, bvh, d_bench_rng);
            cudaEventRecord(ev_stop);
            cudaEventSynchronize(ev_stop);

            float ms = 0.0f;
            cudaEventElapsedTime(&ms, ev_start, ev_stop);
            float mrays = (float)res.w * res.h * BENCH_SPP / 1000000.0f / (ms / 1000.0f);

            std::cout << std::left
                      << std::setw(20) << res.name
                      << std::setw(12) << bench_pixels
                      << std::setw(14) << std::fixed << std::setprecision(1) << ms
                      << std::setw(12) << std::fixed << std::setprecision(1) << mrays
                      << "\n";

            cudaEventDestroy(ev_start);
            cudaEventDestroy(ev_stop);
            cudaFree(d_bench_fb);
            cudaFree(d_bench_rng);
        }
        std::cout << "\n";

        // restaurer la config originale et quitter proprement
        cudaMemcpyToSymbol(d_const_config, &config, sizeof(RenderConfig));
        cudaFree(d_frame_buffer);
        cudaFree(d_rand_states);
        cudaFree(d_materials);
        cudaFree(d_total_luminance);
        cudaStreamDestroy(stream_compute);
        cudaStreamDestroy(stream_transfer);
        bvh_builder.free_gpu_bvh(bvh);
        return 0;
    }

    // -----------------------------------------------
    // Pipeline de rendu GPU en 3 passes :
    //   1. render_kernel     → couleurs HDR brutes
    //   2. compute_avg_luminance → luminance moyenne (reduction parallele)
    //   3. apply_tonemapping_kernel → tone mapping adaptatif + gamma
    // -----------------------------------------------

    if (!args.quiet) std::cout << "Rendu en cours..." << std::flush;
    timer.start("Render Kernel (GPU)");
    cuda_timer.start();
    auto start_time = std::chrono::high_resolution_clock::now();

    render_kernel<<<blocks, threads>>>(d_frame_buffer, bvh, d_rand_states);
    CUDA_SYNC_CHECK();

    cuda_timer.stop();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    timer.stop();

    float kernel_time_ms = cuda_timer.elapsed_ms();

    if (!args.quiet) {
        std::cout << " OK\n";
        std::cout << "Temps rendu: " << duration.count() / 1000.0f << " secondes\n";
        float mrays = (float)config.width * config.height * config.samples_per_pixel / 1000000.0f;
        std::cout << "Performance: " << mrays / (duration.count() / 1000.0f) << " MRays/s\n";
    }

    // --- Reduction parallele : luminance moyenne ---
    timer.start("Luminance Reduction (custom)");
    CudaTimer reduce_timer;
    reduce_timer.start();

    cudaMemset(d_total_luminance, 0, sizeof(float));
    int reduce_block = 256;
    int reduce_grid = (num_pixels + reduce_block - 1) / reduce_block;
    compute_avg_luminance<<<reduce_grid, reduce_block>>>(
        d_frame_buffer, d_total_luminance, num_pixels);
    CUDA_SYNC_CHECK();

    float h_total_lum = 0.0f;
    cudaMemcpy(&h_total_lum, d_total_luminance, sizeof(float), cudaMemcpyDeviceToHost);
    float avg_luminance = h_total_lum / static_cast<float>(num_pixels);

    reduce_timer.stop();
    float custom_reduce_ms = reduce_timer.elapsed_ms();
    timer.stop();

    if (!args.quiet) {
        std::cout << "Luminance moyenne: " << avg_luminance << "\n";
    }

    // --- Reduction Thrust (comparaison librairie) ---
    timer.start("Luminance Reduction (Thrust)");
    CudaTimer thrust_timer;
    thrust_timer.start();

    thrust::device_ptr<Color> thrust_ptr(d_frame_buffer);
    float thrust_total = thrust::transform_reduce(
        thrust_ptr, thrust_ptr + num_pixels,
        LuminanceFunctor(),
        0.0f,
        thrust::plus<float>()
    );
    float thrust_avg = thrust_total / static_cast<float>(num_pixels);

    thrust_timer.stop();
    float thrust_reduce_ms = thrust_timer.elapsed_ms();
    timer.stop();

    // --- Tone mapping adaptatif (second pass) ---
    if (!args.quiet) std::cout << "Tone mapping adaptatif..." << std::flush;
    timer.start("Adaptive Tone Mapping");

    apply_tonemapping_kernel<<<reduce_grid, reduce_block>>>(
        d_frame_buffer, avg_luminance, num_pixels);
    CUDA_SYNC_CHECK();

    timer.stop();
    if (!args.quiet) std::cout << " OK\n";

    // --- Telechargement du framebuffer (pinned memory + stream async) ---
    // cudaMallocHost alloue en page-locked : le DMA GPU peut transferer
    // directement sans staging buffer, permettant un vrai recouvrement
    // avec les operations du stream_compute (cf. Cours 05-Streams).
    timer.start("Framebuffer Download (D2H)");
    Color* h_frame_buffer;
    cudaMallocHost(&h_frame_buffer, num_pixels * sizeof(Color));
    cudaMemcpyAsync(h_frame_buffer, d_frame_buffer,
                    num_pixels * sizeof(Color),
                    cudaMemcpyDeviceToHost, stream_transfer);
    cudaStreamSynchronize(stream_transfer);
    timer.stop();

    timer.start("Save Image");
    if (!args.quiet) std::cout << "Sauvegarde..." << std::flush;
    save_image(args.output_file, h_frame_buffer, config.width, config.height);
    timer.stop();
    if (!args.quiet) std::cout << " OK\n";

    // -----------------------------------------------
    // Profiling detaille
    // -----------------------------------------------
    if (args.profile) {
        timer.print_report("GPU Profiling");
        std::cout << "\n  === Kernel Timing (CUDA events) ===\n";
        std::cout << "  Render kernel:     " << kernel_time_ms << " ms\n";
        std::cout << "  Custom reduction:  " << custom_reduce_ms << " ms\n";
        std::cout << "  Thrust reduction:  " << thrust_reduce_ms << " ms\n";
        std::cout << "  Reduction speedup: x"
                  << thrust_reduce_ms / fmaxf(custom_reduce_ms, 0.001f) << " (Thrust/Custom)\n";
        std::cout << "  Luminance (custom): " << avg_luminance
                  << "  (Thrust): " << thrust_avg << "\n";

        // estimation de la bande passante effective du kernel de rendu
        // chaque pixel : lecture rand_state + ecriture Color = ~52 + 12 = ~64 bytes
        float bytes_rw = (float)num_pixels * 64.0f;
        float bandwidth_gbs = bytes_rw / (kernel_time_ms * 1e6f);
        std::cout << "\n  === Bandwidth Estimation ===\n";
        std::cout << "  Estimated bandwidth: " << bandwidth_gbs << " GB/s\n";

        // occupancy du kernel de rendu
        std::cout << "\n  === Occupancy Analysis ===\n";
        int min_grid_size = 0, best_block_size = 0;
        cudaOccupancyMaxPotentialBlockSize(
            &min_grid_size, &best_block_size,
            render_kernel, 0, 0);
        std::cout << "  render_kernel:\n";
        std::cout << "    Block size used: " << BLOCK_SIZE << "x" << BLOCK_SIZE
                  << " = " << BLOCK_SIZE * BLOCK_SIZE << " threads\n";
        std::cout << "    Optimal block size (API): " << best_block_size << " threads\n";
        std::cout << "    Min grid for full occupancy: " << min_grid_size << " blocks\n";

        int num_blocks_per_sm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &num_blocks_per_sm, render_kernel,
            BLOCK_SIZE * BLOCK_SIZE, 0);

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        int warps_per_block = (BLOCK_SIZE * BLOCK_SIZE) / prop.warpSize;
        int active_warps = num_blocks_per_sm * warps_per_block;
        int max_warps_per_sm = prop.maxThreadsPerMultiProcessor / prop.warpSize;
        float occupancy = 100.0f * active_warps / max_warps_per_sm;

        std::cout << "    Blocks actifs/SM: " << num_blocks_per_sm << "\n";
        std::cout << "    Warps actifs/SM: " << active_warps
                  << " / " << max_warps_per_sm << "\n";
        std::cout << "    Occupancy: " << occupancy << "%\n";

        // occupancy du kernel de reduction
        cudaOccupancyMaxPotentialBlockSize(
            &min_grid_size, &best_block_size,
            compute_avg_luminance, 0, 0);
        std::cout << "\n  compute_avg_luminance:\n";
        std::cout << "    Block size: 256, Optimal (API): "
                  << best_block_size << "\n";

        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &num_blocks_per_sm, compute_avg_luminance, 256, 0);
        active_warps = num_blocks_per_sm * (256 / prop.warpSize);
        occupancy = 100.0f * active_warps / max_warps_per_sm;
        std::cout << "    Blocks actifs/SM: " << num_blocks_per_sm
                  << ", Occupancy: " << occupancy << "%\n";

        std::cout << "\n  === Memory Types Used ===\n";
        std::cout << "  __constant__: Camera (" << sizeof(Camera) << " B)"
                  << " + RenderConfig (" << sizeof(RenderConfig) << " B)\n";
        std::cout << "  __shared__:   256 floats in reduction (" << 256 * sizeof(float) << " B/block)\n";
        std::cout << "  Pinned host:  framebuffer D2H (" << num_pixels * sizeof(Color) << " B)\n";
        std::cout << "  Global:       framebuffer, BVH, rand_states, materials\n";
        std::cout << "  Streams:      2 (compute + transfer, overlapped init)\n";
    }

    // cleanup
    cudaFreeHost(h_frame_buffer);
    cudaFree(d_frame_buffer);
    cudaFree(d_rand_states);
    cudaFree(d_materials);
    cudaFree(d_total_luminance);
    cudaStreamDestroy(stream_compute);
    cudaStreamDestroy(stream_transfer);
    bvh_builder.free_gpu_bvh(bvh);

    return 0;
}
