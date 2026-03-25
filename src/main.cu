/**
 * @file main.cu
 * @brief Point d'entree principal du ray tracer CUDA
 * @details Ce fichier contient la fonction main() qui orchestre l'ensemble du
 *          pipeline de rendu : parsing des arguments, creation de la scene,
 *          construction du BVH, allocation memoire GPU, lancement du rendu
 *          (GPU CUDA ou CPU OpenMP) et sauvegarde de l'image finale.
 *          Il gere egalement le mode interactif avec fenetre OpenGL si active.
 */

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
/**
 * @brief Boucle principale du mode interactif avec rendu progressif en temps reel
 * @details Initialise une fenetre OpenGL, configure l'interop CUDA/OpenGL pour
 *          afficher le rendu directement sur la carte graphique, et entre dans
 *          une boucle de rendu temps reel. La camera est controlable avec les
 *          touches WASD (deplacement), la souris (orientation), Space/Q (monter)
 *          et Ctrl/E (descendre). Le rendu est progressif : les samples s'accumulent
 *          tant que la camera ne bouge pas, jusqu'a max_accumulated_spp.
 *          Le deplacement de la camera reinitialise l'accumulation.
 *          Supporte aussi la capture d'ecran avec la touche P.
 * @param args Arguments en ligne de commande (sensibilite, vitesse, SPP, etc.)
 * @param initial_camera Camera initiale configuree par la scene
 * @param bvh Structure BVH deja construite et uploadee sur le GPU
 * @param config Configuration de rendu (resolution, profondeur, etc.)
 * @param d_rand_states Etats du generateur aleatoire CUDA, deja initialises sur le GPU
 * @return 0 en cas de succes, 1 en cas d'erreur d'initialisation
 */
int run_interactive_mode(
    const Args& args,
    Camera& initial_camera,
    BVH& bvh,
    RenderConfig& config,
    curandState* d_rand_states
) {
    Window window;
    if (!window.initialize(config.width, config.height, "CUDA Ray Tracer - Interactive")) {
        std::cerr << "Failed to initialize window\n";
        return 1;
    }

    GLInterop gl_interop;
    if (!gl_interop.initialize(config.width, config.height)) {
        std::cerr << "Failed to initialize GL interop\n";
        window.cleanup();
        return 1;
    }

    AccumulationBuffer accum_buffer;
    if (!accum_buffer.initialize(config.width, config.height)) {
        std::cerr << "Failed to initialize accumulation buffer\n";
        gl_interop.cleanup();
        window.cleanup();
        return 1;
    }

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

    InputHandler input;
    input.setup_callbacks(window.handle);

    constexpr int BLOCK_SIZE = 16;
    dim3 blocks((config.width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (config.height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    auto last_frame_time = std::chrono::high_resolution_clock::now();
    float fps = 0.0f;
    int frame_count = 0;
    auto fps_timer = std::chrono::high_resolution_clock::now();

    std::cout << "Interactive mode started\n";
    std::cout << "  WASD: Move | Mouse: Look | Space/Q: Up | Ctrl/E: Down\n";
    std::cout << "  Shift: Fast | R: Reset | P: Screenshot | Esc: Quit\n\n";

    while (!window.should_close() && !input.state.should_close) {
        auto current_time = std::chrono::high_resolution_clock::now();
        float delta_time = std::chrono::duration<float>(current_time - last_frame_time).count();
        last_frame_time = current_time;

        input.poll_events();

        bool camera_moved = camera_ctrl.update(input.state, delta_time);

        if (camera_moved || input.state.reset_accumulation) {
            accum_buffer.reset();
        }

        bool should_render = (accum_buffer.get_accumulated_samples() < args.max_accumulated_spp);

        if (should_render) {
            Camera frame_camera = camera_ctrl.build_camera(config.width, config.height);

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

        uchar4* d_display = gl_interop.map_for_cuda();

        convert_to_rgba8<<<blocks, threads>>>(
            d_display,
            accum_buffer.d_buffer,
            config.width, config.height,
            accum_buffer.get_accumulated_samples()
        );
        cudaDeviceSynchronize();

        gl_interop.unmap_from_cuda();

        glClear(GL_COLOR_BUFFER_BIT);
        gl_interop.display();
        window.swap_buffers();

        if (input.state.screenshot_requested) {
            std::vector<Color> h_buffer(config.width * config.height);
            cudaMemcpy(h_buffer.data(), accum_buffer.d_buffer,
                       config.width * config.height * sizeof(Color),
                       cudaMemcpyDeviceToHost);

            int samples = accum_buffer.get_accumulated_samples();
            for (auto& c : h_buffer) {
                if (samples > 0) c = c / static_cast<float>(samples);
                c = Color(c.x / (1.0f + c.x), c.y / (1.0f + c.y), c.z / (1.0f + c.z));
                c = Color(powf(fmaxf(0.0f, c.x), 1.0f/2.2f),
                          powf(fmaxf(0.0f, c.y), 1.0f/2.2f),
                          powf(fmaxf(0.0f, c.z), 1.0f/2.2f));
            }

            save_image("output/screenshot.png", h_buffer.data(), config.width, config.height);
            std::cout << "Screenshot saved: output/screenshot.png (" << samples << " SPP)\n";
        }

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

    accum_buffer.cleanup();
    gl_interop.cleanup();
    window.cleanup();

    return 0;
}
#endif

/**
 * @brief Recupere le pointeur vers le materiau d'un objet hittable
 * @details Selon le type de l'objet (sphere ou plan), retourne le pointeur
 *          vers le materiau associe. Utile notamment lors de la conversion
 *          des pointeurs host vers device pour le transfert GPU.
 * @param obj Reference constante vers l'objet dont on veut le materiau
 * @return Pointeur vers le Material de l'objet, ou nullptr si le type est inconnu
 */
inline Material* get_material_ptr(const HittableObject& obj) {
    switch (obj.type) {
        case HittableType::SPHERE: return obj.data.sphere.mat;
        case HittableType::PLANE:  return obj.data.plane.mat;
        default:                   return nullptr;
    }
}

/**
 * @brief Modifie le pointeur materiau d'un objet hittable
 * @details Remplace le pointeur materiau de l'objet par un nouveau pointeur.
 *          Utilise principalement pour la "fixup" des pointeurs lors du transfert
 *          des donnees du CPU vers le GPU : les pointeurs host doivent etre
 *          remplaces par les pointeurs device correspondants.
 * @param obj Reference vers l'objet dont on veut modifier le materiau
 * @param mat Nouveau pointeur vers le materiau (typiquement un pointeur device)
 */
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

/**
 * @brief Point d'entree principal du ray tracer
 * @details Orchestre l'ensemble du pipeline de rendu en plusieurs etapes :
 *          1. Parsing des arguments en ligne de commande
 *          2. Creation de la scene par defaut (camera, objets, materiaux)
 *          3. Selon le mode choisi (CPU ou GPU) :
 *             - CPU : construction du BVH cote host, rendu OpenMP
 *             - GPU : allocation memoire CUDA, upload des materiaux, fixup des
 *               pointeurs host->device, construction et upload du BVH, initialisation
 *               des etats aleatoires, puis lancement du kernel de rendu
 *          4. Sauvegarde de l'image finale
 *          En mode interactif, delegue a run_interactive_mode() apres l'initialisation GPU.
 * @param argc Nombre d'arguments de la ligne de commande
 * @param argv Tableau des arguments de la ligne de commande
 * @return 0 en cas de succes, 1 en cas d'erreur
 */
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
