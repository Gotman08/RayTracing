#ifndef RAYTRACER_RENDERING_CPU_RENDERER_CUH
#define RAYTRACER_RENDERING_CPU_RENDERER_CUH

/**
 * @file cpu_renderer.cuh
 * @brief Moteur de rendu CPU du raytracer, parallelise avec OpenMP
 * @details Ce fichier contient la version CPU du moteur de rendu. L'algorithme est
 *          identique a la version GPU (ray_color + multi-echantillonnage + tone mapping),
 *          mais il utilise CPURandom au lieu de curand et OpenMP pour la parallelisation.
 *          Chaque thread OpenMP traite une ligne complete de l'image.
 */

#include <omp.h>
#include <vector>
#include <cmath>

#include "raytracer/core/vec3.cuh"
#include "raytracer/core/ray.cuh"
#include "raytracer/core/random.cuh"
#include "raytracer/core/hit_record.cuh"
#include "raytracer/camera/camera.cuh"
#include "raytracer/geometry/hittable_list.cuh"
#include "raytracer/acceleration/bvh.cuh"
#include "raytracer/rendering/render_config.cuh"
#include "raytracer/rendering/material_dispatch.cuh"
#include "raytracer/rendering/tone_mapping.cuh"

namespace rt {

/**
 * @brief Calcule la couleur d'un rayon sur CPU par rebonds successifs
 * @details Version CPU de ray_color(). L'algorithme est le meme : on suit le rayon
 *          de rebond en rebond, en accumulant l'attenuation de chaque materiau.
 *          Si le rayon ne touche rien, on retourne la couleur du ciel ou du fond.
 *          La seule difference est l'utilisation de CPURandom au lieu de curand
 *          et de scatter_cpu() au lieu de scatter().
 * @param initial_ray Le rayon initial lance depuis la camera
 * @param bvh La structure d'acceleration BVH contenant la scene
 * @param max_depth Le nombre maximal de rebonds autorises
 * @param config La configuration du rendu (ciel, fond, etc.)
 * @param rng Le generateur de nombres aleatoires CPU pour ce thread
 * @return La couleur finale calculee pour ce rayon
 */
inline Color ray_color_cpu(
    const Ray& initial_ray,
    const BVH& bvh,
    int max_depth,
    const RenderConfig& config,
    CPURandom& rng
) {
    Color accumulated(0, 0, 0);
    Color attenuation(1, 1, 1);
    Ray current_ray = initial_ray;

    for (int depth = 0; depth < max_depth; depth++) {
        HitRecord rec;

        if (bvh.hit(current_ray, Interval(0.001f, 1e30f), rec)) {
            Ray scattered;
            Color scatter_attenuation;

            if (scatter_cpu(*rec.mat, current_ray, rec, scatter_attenuation, scattered, rng)) {
                attenuation = attenuation * scatter_attenuation;
                current_ray = scattered;
            } else {
                break;
            }
        } else {
            if (config.use_sky) {
                accumulated += attenuation * config.sky.get_color(current_ray.direction());
            } else {
                accumulated += attenuation * config.background;
            }
            break;
        }
    }

    return accumulated;
}

/**
 * @brief Fonction de rendu CPU parallelisee avec OpenMP
 * @details Cette fonction parcourt tous les pixels de l'image en parallele grace
 *          a OpenMP. Chaque thread gere une ligne complete de l'image et possede
 *          son propre generateur aleatoire (CPURandom) pour eviter les conflits.
 *          Pour chaque pixel, on lance plusieurs rayons (multi-echantillonnage),
 *          on fait la moyenne, puis on applique le tone mapping de Reinhard et
 *          la correction gamma avant de clamper dans [0, 1].
 * @param frame_buffer Le buffer de sortie contenant les couleurs des pixels
 * @param camera La camera utilisee pour generer les rayons
 * @param bvh La structure d'acceleration BVH de la scene
 * @param config La configuration du rendu (resolution, samples, profondeur, etc.)
 */
inline void render_cpu(
    Color* frame_buffer,
    const Camera& camera,
    const BVH& bvh,
    const RenderConfig& config
) {
    const int width = config.width;
    const int height = config.height;
    const int samples = config.samples_per_pixel;
    const int max_depth = config.max_depth;

    #pragma omp parallel for schedule(dynamic, 1) collapse(1)
    for (int j = 0; j < height; j++) {
        CPURandom rng(j * 1337 + 42);

        for (int i = 0; i < width; i++) {
            Color pixel_color(0, 0, 0);

            for (int s = 0; s < samples; s++) {
                Ray r = camera.get_ray_cpu(i, j, rng);
                pixel_color += ray_color_cpu(r, bvh, max_depth, config, rng);
            }

            pixel_color = pixel_color / static_cast<float>(samples);

            pixel_color = apply_tone_mapping(pixel_color);
            pixel_color = gamma_correct(pixel_color);

            pixel_color.x = fminf(1.0f, fmaxf(0.0f, pixel_color.x));
            pixel_color.y = fminf(1.0f, fmaxf(0.0f, pixel_color.y));
            pixel_color.z = fminf(1.0f, fmaxf(0.0f, pixel_color.z));

            int pixel_index = j * width + i;
            frame_buffer[pixel_index] = pixel_color;
        }
    }
}

}

#endif
