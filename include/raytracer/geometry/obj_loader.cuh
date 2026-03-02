#ifndef RAYTRACER_GEOMETRY_OBJ_LOADER_CUH
#define RAYTRACER_GEOMETRY_OBJ_LOADER_CUH

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "raytracer/geometry/triangle.cuh"
#include "raytracer/geometry/hittable_list.cuh"

namespace rt {

// Simple OBJ loader - loads triangulated meshes
class OBJLoader {
public:
    std::vector<Point3> vertices;
    std::vector<Vec3> normals;
    std::vector<std::array<int, 3>> faces;       // vertex indices
    std::vector<std::array<int, 3>> face_normals; // normal indices

    bool load(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            return false;
        }

        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string prefix;
            iss >> prefix;

            if (prefix == "v") {
                // Vertex position
                float x, y, z;
                iss >> x >> y >> z;
                vertices.push_back(Point3(x, y, z));
            }
            else if (prefix == "vn") {
                // Vertex normal
                float x, y, z;
                iss >> x >> y >> z;
                normals.push_back(Vec3(x, y, z).normalized());
            }
            else if (prefix == "f") {
                // Face (triangle only - assumes triangulated mesh)
                std::array<int, 3> face_verts = {0, 0, 0};
                std::array<int, 3> face_norms = {-1, -1, -1};

                for (int i = 0; i < 3; i++) {
                    std::string vertex_data;
                    iss >> vertex_data;

                    // Parse v/vt/vn or v//vn or v/vt or v format
                    int v_idx = 0, vt_idx = 0, vn_idx = 0;

                    size_t slash1 = vertex_data.find('/');
                    if (slash1 == std::string::npos) {
                        // Just vertex index
                        v_idx = std::stoi(vertex_data);
                    } else {
                        v_idx = std::stoi(vertex_data.substr(0, slash1));

                        size_t slash2 = vertex_data.find('/', slash1 + 1);
                        if (slash2 != std::string::npos) {
                            // Has texture and/or normal
                            std::string vt_str = vertex_data.substr(slash1 + 1, slash2 - slash1 - 1);
                            if (!vt_str.empty()) {
                                vt_idx = std::stoi(vt_str);
                            }
                            std::string vn_str = vertex_data.substr(slash2 + 1);
                            if (!vn_str.empty()) {
                                vn_idx = std::stoi(vn_str);
                            }
                        } else {
                            // Just v/vt
                            std::string vt_str = vertex_data.substr(slash1 + 1);
                            if (!vt_str.empty()) {
                                vt_idx = std::stoi(vt_str);
                            }
                        }
                    }

                    // OBJ indices are 1-based, convert to 0-based
                    face_verts[i] = v_idx - 1;
                    face_norms[i] = vn_idx > 0 ? vn_idx - 1 : -1;
                }

                faces.push_back(face_verts);
                face_normals.push_back(face_norms);
            }
        }

        return !vertices.empty() && !faces.empty();
    }

    // Add triangles to the object array
    int add_to_scene(
        HittableObject* objects,
        int& obj_count,
        int max_objects,
        Material* mat,
        const Point3& offset = Point3(0, 0, 0),
        float scale = 1.0f
    ) {
        int triangles_added = 0;

        for (size_t i = 0; i < faces.size(); i++) {
            if (obj_count >= max_objects) {
                break;
            }

            const auto& face = faces[i];
            const auto& fnorm = face_normals[i];

            Point3 v0 = vertices[face[0]] * scale + offset;
            Point3 v1 = vertices[face[1]] * scale + offset;
            Point3 v2 = vertices[face[2]] * scale + offset;

            objects[obj_count].type = HittableType::TRIANGLE;
            objects[obj_count].data.triangle = Triangle(v0, v1, v2, mat);

            // Apply smooth shading if normals are available
            if (fnorm[0] >= 0 && fnorm[1] >= 0 && fnorm[2] >= 0 &&
                fnorm[0] < (int)normals.size() &&
                fnorm[1] < (int)normals.size() &&
                fnorm[2] < (int)normals.size()) {
                objects[obj_count].data.triangle.set_normals(
                    normals[fnorm[0]],
                    normals[fnorm[1]],
                    normals[fnorm[2]]
                );
            }

            objects[obj_count].bbox = objects[obj_count].data.triangle.bounding_box();
            obj_count++;
            triangles_added++;
        }

        return triangles_added;
    }

    void clear() {
        vertices.clear();
        normals.clear();
        faces.clear();
        face_normals.clear();
    }
};

} // namespace rt

#endif // RAYTRACER_GEOMETRY_OBJ_LOADER_CUH
