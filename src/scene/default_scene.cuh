/**
 * @file default_scene.cuh
 * @brief Creation de la scene par defaut du ray tracer
 * @details Ce fichier contient la fonction qui genere une scene originale
 *          sur le theme d'un systeme solaire stylise. La scene comprend un sol
 *          sombre, un soleil central metallique dore, plusieurs planetes de
 *          differents materiaux (verre, metal, diffus), un anneau orbital
 *          de micro-spheres et des etoiles en arriere-plan.
 */

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

/**
 * @brief Ajoute une sphere a la scene avec son materiau
 * @details Fonction utilitaire qui simplifie l'ajout d'un objet sphere dans
 *          les tableaux d'objets et de materiaux. Elle cree la sphere, calcule
 *          sa boite englobante et incremente les compteurs automatiquement.
 * @param objects Tableau d'objets de la scene
 * @param materials Tableau de materiaux de la scene
 * @param obj_count Compteur d'objets (incremente automatiquement)
 * @param mat_count Compteur de materiaux (incremente automatiquement)
 * @param center Position du centre de la sphere
 * @param radius Rayon de la sphere
 * @param mat Materiau a assigner a la sphere
 */
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

/**
 * @brief Cree la scene "Systeme Solaire Stylise"
 * @details Construit une scene spatiale originale avec :
 *          - Un sol sombre qui simule une surface cosmique
 *          - Un soleil central en metal dore avec une bulle de verre interne
 *          - Plusieurs planetes de tailles et materiaux varies (verre, metal, diffus)
 *          - Un anneau orbital de micro-spheres autour du soleil
 *          - Des etoiles metalliques dispersees en arriere-plan
 *          La camera est positionnee en vue plongeante cinematique.
 * @param camera Reference vers la camera a initialiser
 * @param world Reference vers la liste d'objets de la scene
 * @param objects Tableau pre-alloue pour stocker les objets HittableObject
 * @param materials Tableau pre-alloue pour stocker les materiaux Material
 * @param obj_count Compteur d'objets, incremente au fur et a mesure (entree/sortie)
 * @param mat_count Compteur de materiaux, incremente au fur et a mesure (entree/sortie)
 * @param config Configuration de rendu, utilisee pour la resolution et le ciel
 */
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
        Color(0.02f, 0.02f, 0.06f),
        Color(0.0f, 0.0f, 0.02f)
    );
    config.sky.set_sun(Vec3(1.0f, 0.8f, 0.3f), 3.0f, 0.015f);

    add_sphere(objects, materials, obj_count, mat_count,
        Point3(0, -1000, 0), 1000.0f,
        create_lambertian(Color(0.05f, 0.05f, 0.08f)));

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
