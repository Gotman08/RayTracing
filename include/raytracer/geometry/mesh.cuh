#ifndef RAYTRACER_GEOMETRY_MESH_CUH
#define RAYTRACER_GEOMETRY_MESH_CUH

#include "raytracer/geometry/triangle.cuh"

namespace rt {

// Simple mesh structure holding triangles
// For large meshes, use BVH acceleration
struct Mesh {
    Triangle* triangles;
    int num_triangles;
    AABB bbox;
    Material* mat;

    __host__ __device__ Mesh() : triangles(nullptr), num_triangles(0), mat(nullptr) {}

    __host__ __device__ void compute_bounding_box() {
        if (num_triangles == 0) return;

        bbox = triangles[0].bounding_box();
        for (int i = 1; i < num_triangles; i++) {
            bbox = AABB(bbox, triangles[i].bounding_box());
        }
    }

    __host__ __device__ bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const {
        // Quick AABB test first
        if (!bbox.hit(r, ray_t))
            return false;

        HitRecord temp_rec;
        bool hit_anything = false;
        float closest_so_far = ray_t.max;

        for (int i = 0; i < num_triangles; i++) {
            if (triangles[i].hit(r, Interval(ray_t.min, closest_so_far), temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }

    __host__ __device__ const AABB& bounding_box() const { return bbox; }
};

} // namespace rt

#endif // RAYTRACER_GEOMETRY_MESH_CUH
