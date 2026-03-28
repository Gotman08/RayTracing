/** @file default_scene.cuh
 * @brief Scene par defaut : systeme solaire stylise */

#ifndef RAYTRACER_SCENE_DEFAULT_SCENE_CUH
#define RAYTRACER_SCENE_DEFAULT_SCENE_CUH

#include <cstdlib>
#include <cmath>
#include "raytracer/core/vec3.cuh"
#include "raytracer/camera/camera.cuh"
#include "raytracer/geometry/hittable_list.cuh"
#include "raytracer/geometry/sphere.cuh"
#include "raytracer/materials/lambertian.cuh"
#include "raytracer/materials/metal.cuh"
#include "raytracer/materials/dielectric.cuh"
#include "raytracer/rendering/render_config.cuh"

namespace rt {

/** @brief Ajoute une sphere + son materiau dans les tableaux de la scene */
inline void add_sphere(
    HittableObject* objects, Material* materials,
    int& obj_count, int& mat_count,
    const Point3& center, float radius, const Material& mat
) {
    materials[mat_count] = mat;
    objects[obj_count].type = HittableType::SPHERE;
    objects[obj_count].data.sphere = Sphere(center, radius, &materials[mat_count]);
    objects[obj_count].bbox = objects[obj_count].data.sphere.bounding_box();
    mat_count++;
    obj_count++;
}

// -----------------------------------------------
// Composition exacte de la scene par defaut
// Ces constantes permettent une allocation dynamique precise dans main.cu,
// eliminant le magic number "estimated_objects = 200".
// -----------------------------------------------

constexpr int SCENE_FIXED_OBJECTS = 10;  ///< sol + soleil + planetes
constexpr int SCENE_RING_COUNT    = 40;
constexpr int SCENE_STAR_COUNT    = 30;
constexpr int SCENE_TOTAL         = SCENE_FIXED_OBJECTS + SCENE_RING_COUNT + SCENE_STAR_COUNT;

/** @brief Nb d'objets total + marge pour alloc */
inline constexpr int count_scene_objects() {
    return SCENE_TOTAL + 10;  // 80 objets exacts + 10 de marge = 90
}

/** @brief Construit la scene solaire : sol, soleil, planetes, anneau, etoiles */
inline void create_default_scene(
    Camera& camera,
    HittableList& world,
    HittableObject* objects,
    Material* materials,
    int& obj_count,
    int& mat_count,
    RenderConfig& config
) {
    camera.initialize(
        config.width, config.height,
        Point3(8, 6, 12),
        Point3(0, 1, 0),
        Vec3(0, 1, 0),
        30.0f, 0.05f, 14.0f
    );

    config.use_sky = true;
    config.sky = Sky(
        Color(0.35f, 0.35f, 0.45f),
        Color(0.08f, 0.08f, 0.2f)
    );
    config.sky.set_sun(Vec3(1.0f, 0.8f, 0.3f), 8.0f, 0.05f);

    add_sphere(objects, materials, obj_count, mat_count,
        Point3(0, -1000, 0), 1000.0f,
        create_lambertian(Color(0.15f, 0.15f, 0.2f)));

    add_sphere(objects, materials, obj_count, mat_count,
        Point3(0, 2, 0), 2.0f,
        create_metal(Color(0.95f, 0.7f, 0.2f), 0.0f));

    add_sphere(objects, materials, obj_count, mat_count,
        Point3(0, 2, 0), 1.5f,
        create_dielectric(1.5f));

    add_sphere(objects, materials, obj_count, mat_count,
        Point3(4.0f, 1.2f, -1.0f), 0.8f,
        create_dielectric(1.7f));

    add_sphere(objects, materials, obj_count, mat_count,
        Point3(-3.0f, 0.8f, 2.0f), 0.6f,
        create_lambertian(Color(0.8f, 0.15f, 0.1f)));

    add_sphere(objects, materials, obj_count, mat_count,
        Point3(-1.5f, 1.0f, -4.0f), 0.7f,
        create_metal(Color(0.3f, 0.4f, 0.9f), 0.1f));

    add_sphere(objects, materials, obj_count, mat_count,
        Point3(2.5f, 0.7f, 3.5f), 0.5f,
        create_lambertian(Color(0.5f, 0.1f, 0.6f)));

    add_sphere(objects, materials, obj_count, mat_count,
        Point3(4.8f, 1.8f, -0.5f), 0.25f,
        create_metal(Color(0.9f, 0.9f, 0.95f), 0.0f));

    add_sphere(objects, materials, obj_count, mat_count,
        Point3(-4.0f, 0.75f, -2.5f), 0.55f,
        create_lambertian(Color(0.1f, 0.6f, 0.2f)));

    add_sphere(objects, materials, obj_count, mat_count,
        Point3(1.0f, 0.5f, -3.0f), 0.4f,
        create_metal(Color(0.75f, 0.3f, 0.15f), 0.05f));

    srand(77);
    const float ring_radius = 5.5f;
    const float ring_y = 2.0f;
    const int ring_count = 40;

    for (int i = 0; i < ring_count; i++) {
        float angle = (float)i / ring_count * 2.0f * 3.14159265f;
        float r_offset = 0.3f * ((float)rand() / RAND_MAX - 0.5f);
        float y_offset = 0.15f * ((float)rand() / RAND_MAX - 0.5f);
        float sphere_r = 0.08f + 0.04f * (float)rand() / RAND_MAX;

        Point3 pos(
            (ring_radius + r_offset) * cosf(angle),
            ring_y + y_offset,
            (ring_radius + r_offset) * sinf(angle)
        );

        float mat_choice = (float)rand() / RAND_MAX;
        Material mat;
        if (mat_choice < 0.4f) {
            mat = create_metal(
                Color(0.8f + 0.2f * (float)rand() / RAND_MAX,
                      0.8f + 0.2f * (float)rand() / RAND_MAX,
                      0.8f + 0.2f * (float)rand() / RAND_MAX),
                0.0f);
        } else if (mat_choice < 0.7f) {
            mat = create_dielectric(1.5f);
        } else {
            mat = create_lambertian(
                Color((float)rand() / RAND_MAX * 0.8f + 0.2f,
                      (float)rand() / RAND_MAX * 0.8f + 0.2f,
                      (float)rand() / RAND_MAX * 0.8f + 0.2f));
        }

        add_sphere(objects, materials, obj_count, mat_count,
            pos, sphere_r, mat);
    }

    const int star_count = 30;
    for (int i = 0; i < star_count; i++) {
        float x = 20.0f * ((float)rand() / RAND_MAX - 0.5f);
        float y = 5.0f + 10.0f * (float)rand() / RAND_MAX;
        float z = 20.0f * ((float)rand() / RAND_MAX - 0.5f);
        float star_r = 0.03f + 0.03f * (float)rand() / RAND_MAX;

        add_sphere(objects, materials, obj_count, mat_count,
            Point3(x, y, z), star_r,
            create_metal(Color(0.95f, 0.95f, 1.0f), 0.0f));
    }
}

}

#endif
