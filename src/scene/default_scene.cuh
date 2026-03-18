/**
 * @file default_scene.cuh
 * @brief Creation de la scene par defaut du ray tracer
 * @details Ce fichier contient la fonction qui genere la scene de demonstration
 *          inspiree du livre "Ray Tracing in One Weekend" de Peter Shirley.
 *          La scene comprend un sol, trois grandes spheres centrales et un
 *          ensemble de petites spheres aleatoires avec differents materiaux.
 */

#ifndef RAYTRACER_SCENE_DEFAULT_SCENE_CUH
#define RAYTRACER_SCENE_DEFAULT_SCENE_CUH

#include <cstdlib>
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
 * @brief Cree la scene par defaut avec sol, spheres principales et spheres aleatoires
 * @details Initialise la camera avec un point de vue classique (position 13,2,3 regardant
 *          l'origine) et construit la scene suivante :
 *          - Un sol gris (grande sphere de rayon 1000, materiau lambertien)
 *          - Une sphere dielectrique (verre, indice 1.5) au centre
 *          - Une sphere lambertienne (diffuse, marron) a gauche
 *          - Une sphere metallique (reflechissante, doree) a droite
 *          - Des petites spheres aleatoires (lambertien, metal ou verre) dans une grille
 *          La graine aleatoire est fixee a 42 pour la reproductibilite.
 * @param camera Reference vers la camera a initialiser
 * @param world Reference vers la liste d'objets de la scene (non utilisee directement ici)
 * @param objects Tableau pre-alloue pour stocker les objets HittableObject
 * @param materials Tableau pre-alloue pour stocker les materiaux Material
 * @param obj_count Compteur d'objets, incremente au fur et a mesure de l'ajout (entree/sortie)
 * @param mat_count Compteur de materiaux, incremente au fur et a mesure de l'ajout (entree/sortie)
 * @param config Configuration de rendu, utilisee pour la resolution et le flag use_sky
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
        Point3(13, 2, 3),
        Point3(0, 0, 0),
        Vec3(0, 1, 0),
        20.0f, 0.1f, 10.0f
    );

    materials[mat_count] = create_lambertian(Color(0.5f, 0.5f, 0.5f));
    objects[obj_count].type = HittableType::SPHERE;
    objects[obj_count].data.sphere = Sphere(Point3(0, -1000, 0), 1000, &materials[mat_count]);
    objects[obj_count].bbox = objects[obj_count].data.sphere.bounding_box();
    mat_count++; obj_count++;

    materials[mat_count] = create_dielectric(1.5f);
    objects[obj_count].type = HittableType::SPHERE;
    objects[obj_count].data.sphere = Sphere(Point3(0, 1, 0), 1.0f, &materials[mat_count]);
    objects[obj_count].bbox = objects[obj_count].data.sphere.bounding_box();
    mat_count++; obj_count++;

    materials[mat_count] = create_lambertian(Color(0.4f, 0.2f, 0.1f));
    objects[obj_count].type = HittableType::SPHERE;
    objects[obj_count].data.sphere = Sphere(Point3(-4, 1, 0), 1.0f, &materials[mat_count]);
    objects[obj_count].bbox = objects[obj_count].data.sphere.bounding_box();
    mat_count++; obj_count++;

    materials[mat_count] = create_metal(Color(0.7f, 0.6f, 0.5f), 0.0f);
    objects[obj_count].type = HittableType::SPHERE;
    objects[obj_count].data.sphere = Sphere(Point3(4, 1, 0), 1.0f, &materials[mat_count]);
    objects[obj_count].bbox = objects[obj_count].data.sphere.bounding_box();
    mat_count++; obj_count++;

    srand(42);
    for (int a = -5; a < 5; a++) {
        for (int b = -5; b < 5; b++) {
            float choose_mat = (float)rand() / RAND_MAX;
            Point3 center(a + 0.9f * (float)rand() / RAND_MAX, 0.2f, b + 0.9f * (float)rand() / RAND_MAX);

            if ((center - Point3(4, 0.2f, 0)).length() > 0.9f) {
                if (choose_mat < 0.6f) {
                    Color albedo(
                        (float)rand() / RAND_MAX * (float)rand() / RAND_MAX,
                        (float)rand() / RAND_MAX * (float)rand() / RAND_MAX,
                        (float)rand() / RAND_MAX * (float)rand() / RAND_MAX
                    );
                    materials[mat_count] = create_lambertian(albedo);
                } else if (choose_mat < 0.85f) {
                    Color albedo(
                        0.5f + 0.5f * (float)rand() / RAND_MAX,
                        0.5f + 0.5f * (float)rand() / RAND_MAX,
                        0.5f + 0.5f * (float)rand() / RAND_MAX
                    );
                    float fuzz = 0.5f * (float)rand() / RAND_MAX;
                    materials[mat_count] = create_metal(albedo, fuzz);
                } else {
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

}

#endif
