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
