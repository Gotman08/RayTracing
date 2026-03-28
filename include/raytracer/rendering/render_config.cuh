#ifndef RAYTRACER_RENDERING_RENDER_CONFIG_CUH
#define RAYTRACER_RENDERING_RENDER_CONFIG_CUH

/** @file render_config.cuh
 * @brief Cfg du rendu : resolution, samples, profondeur, ciel/fond */

#include "raytracer/core/vec3.cuh"
#include "raytracer/environment/sky.cuh"

namespace rt {

/** @brief Params de rendu (resolution, sampling, profondeur, environnement) */
struct RenderConfig {
    int width;
    int height;
    int samples_per_pixel;
    int max_depth;
    bool use_sky;
    Sky sky;
    Color background;

    /** @brief Ctor par defaut : 800x600, 100 spp, profondeur 50 */
    RenderConfig()
        : width(800), height(600), samples_per_pixel(100), max_depth(50),
          use_sky(true), background(0, 0, 0) {
        sky = Sky(Color(1, 1, 1), Color(0.5f, 0.7f, 1.0f));
    }
};

}

#endif
