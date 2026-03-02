/**
 * Scene Loading from JSON
 */

#ifndef RAYTRACER_IO_SCENE_LOADER_CUH
#define RAYTRACER_IO_SCENE_LOADER_CUH

#include <iostream>
#include <fstream>
#include <string>
#include "json.hpp"

#include "raytracer/core/vec3.cuh"
#include "raytracer/camera/camera.cuh"
#include "raytracer/geometry/hittable_list.cuh"
#include "raytracer/geometry/sphere.cuh"
#include "raytracer/geometry/quad.cuh"
#include "raytracer/geometry/box.cuh"
#include "raytracer/geometry/triangle.cuh"
#include "raytracer/geometry/plane.cuh"
#include "raytracer/geometry/obj_loader.cuh"
#include "raytracer/materials/lambertian.cuh"
#include "raytracer/materials/metal.cuh"
#include "raytracer/materials/dielectric.cuh"
#include "raytracer/materials/emissive.cuh"
#include "raytracer/rendering/renderer.cuh"
#include "utils/json_helpers.cuh"

using json = nlohmann::json;

namespace rt {

inline Material* create_material_from_json(const json& mat_json, Material* materials, int& mat_count) {
    std::string type = mat_json.value("type", "lambertian");
    Material* mat = &materials[mat_count++];

    if (type == "lambertian") {
        *mat = create_lambertian(json_to_color(mat_json, "color", Color(0.5f, 0.5f, 0.5f)));
    } else if (type == "metal") {
        Color c = json_to_color(mat_json, "color", Color(0.8f, 0.8f, 0.8f));
        *mat = create_metal(c, mat_json.value("fuzz", 0.0f));
    } else if (type == "dielectric" || type == "glass") {
        *mat = create_dielectric(mat_json.value("ior", 1.5f));
    } else if (type == "emissive" || type == "light") {
        Color c = json_to_color(mat_json, "color", Color(1.0f, 1.0f, 1.0f));
        *mat = create_emissive(c, mat_json.value("strength", 1.0f));
    } else if (type == "checker") {
        Color c1 = json_to_color(mat_json, "color1", Color(0.2f, 0.3f, 0.1f));
        Color c2 = json_to_color(mat_json, "color2", Color(0.9f, 0.9f, 0.9f));
        *mat = create_lambertian_checker(mat_json.value("scale", 10.0f), c1, c2);
    } else if (type == "noise" || type == "perlin") {
        Color c = json_to_color(mat_json, "color", Color(1.0f, 1.0f, 1.0f));
        *mat = create_lambertian_noise(mat_json.value("scale", 4.0f), c, mat_json.value("seed", 42));
    } else {
        *mat = create_lambertian(Color(0.5f, 0.5f, 0.5f));
    }

    return mat;
}

inline bool load_scene_from_json(
    const std::string& filename,
    Camera& camera,
    HittableList& world,
    HittableObject* objects,
    Material* materials,
    int& obj_count,
    int& mat_count,
    RenderConfig& config,
    int max_objects,
    int max_materials
) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open scene file: " << filename << std::endl;
        return false;
    }

    json scene;
    try {
        scene = json::parse(file);
    } catch (const json::parse_error& e) {
        std::cerr << "Error: Invalid JSON in scene file: " << filename << std::endl;
        std::cerr << "  " << e.what() << std::endl;
        return false;
    }

    // Camera
    if (scene.contains("camera")) {
        auto& cam = scene["camera"];
        camera.initialize(
            config.width, config.height,
            json_to_point3(cam, "lookfrom", Point3(13, 2, 3)),
            json_to_point3(cam, "lookat", Point3(0, 0, 0)),
            json_to_vec3(cam, "vup", Vec3(0, 1, 0)),
            cam.value("fov", 20.0f),
            cam.value("aperture", 0.0f),
            cam.value("focus_dist", 10.0f)
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
            if (obj_count >= max_objects) {
                std::cerr << "Warning: Maximum object limit (" << max_objects << ") reached, skipping remaining objects" << std::endl;
                break;
            }
            if (mat_count >= max_materials) {
                std::cerr << "Warning: Maximum material limit (" << max_materials << ") reached, skipping remaining objects" << std::endl;
                break;
            }

            std::string type = obj.value("type", "sphere");

            Material* mat = nullptr;
            if (obj.contains("material")) {
                mat = create_material_from_json(obj["material"], materials, mat_count);
            } else {
                mat = &materials[mat_count++];
                *mat = create_lambertian(Color(0.5f, 0.5f, 0.5f));
            }

            if (type == "sphere") {
                objects[obj_count].type = HittableType::SPHERE;
                objects[obj_count].data.sphere = Sphere(
                    json_to_point3(obj, "center", Point3(0, 0, 0)),
                    obj.value("radius", 1.0f), mat
                );
                objects[obj_count].bbox = objects[obj_count].data.sphere.bounding_box();
                obj_count++;

            } else if (type == "quad") {
                objects[obj_count].type = HittableType::QUAD;
                objects[obj_count].data.quad = Quad(
                    json_to_point3(obj, "Q", Point3(0, 0, 0)),
                    json_to_vec3(obj, "u", Vec3(1, 0, 0)),
                    json_to_vec3(obj, "v", Vec3(0, 1, 0)),
                    mat
                );
                objects[obj_count].bbox = objects[obj_count].data.quad.bounding_box();
                obj_count++;

            } else if (type == "box") {
                objects[obj_count].type = HittableType::BOX;
                objects[obj_count].data.box = Box(
                    json_to_point3(obj, "min", Point3(0, 0, 0)),
                    json_to_point3(obj, "max", Point3(1, 1, 1)),
                    mat
                );
                objects[obj_count].bbox = objects[obj_count].data.box.bounding_box();
                obj_count++;

            } else if (type == "triangle") {
                objects[obj_count].type = HittableType::TRIANGLE;
                objects[obj_count].data.triangle = Triangle(
                    json_to_point3(obj, "v0", Point3(0, 0, 0)),
                    json_to_point3(obj, "v1", Point3(1, 0, 0)),
                    json_to_point3(obj, "v2", Point3(0, 1, 0)),
                    mat
                );

                if (obj.contains("n0") && obj.contains("n1") && obj.contains("n2")) {
                    objects[obj_count].data.triangle.set_normals(
                        json_to_vec3(obj, "n0", Vec3(0, 0, 0)),
                        json_to_vec3(obj, "n1", Vec3(0, 0, 0)),
                        json_to_vec3(obj, "n2", Vec3(0, 0, 0))
                    );
                }

                objects[obj_count].bbox = objects[obj_count].data.triangle.bounding_box();
                obj_count++;

            } else if (type == "plane") {
                objects[obj_count].type = HittableType::PLANE;
                objects[obj_count].data.plane = Plane(
                    json_to_point3(obj, "point", Point3(0, 0, 0)),
                    json_to_vec3(obj, "normal", Vec3(0, 1, 0)),
                    mat
                );
                objects[obj_count].bbox = objects[obj_count].data.plane.bounding_box();
                obj_count++;

            } else if (type == "mesh" || type == "obj") {
                std::string mesh_file = obj.value("file", "");
                if (mesh_file.empty()) {
                    std::cerr << "Warning: mesh object missing 'file' field, skipping" << std::endl;
                    mat_count--;
                    continue;
                }

                OBJLoader loader;
                if (loader.load(mesh_file)) {
                    int triangles_added = loader.add_to_scene(
                        objects, obj_count, max_objects, mat,
                        json_to_point3(obj, "offset", Point3(0, 0, 0)),
                        obj.value("scale", 1.0f)
                    );
                    std::cout << "Loaded mesh: " << mesh_file << " (" << triangles_added << " triangles)" << std::endl;
                } else {
                    std::cerr << "Warning: Failed to load mesh file: " << mesh_file << std::endl;
                    mat_count--;
                }
            }
        }
    }

    return true;
}

} // namespace rt

#endif // RAYTRACER_IO_SCENE_LOADER_CUH
