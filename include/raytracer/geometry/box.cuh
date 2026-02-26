#ifndef RAYTRACER_GEOMETRY_BOX_CUH
#define RAYTRACER_GEOMETRY_BOX_CUH

#include "raytracer/geometry/hittable.cuh"
#include "raytracer/geometry/quad.cuh"

namespace rt {

class Box : public Hittable {
public:
    Quad sides[6];
    int num_sides;
    Material* mat;

    __host__ __device__ Box() : Hittable(HittableType::BOX), num_sides(0), mat(nullptr) {}

    __host__ __device__ Box(const Point3& a, const Point3& b, Material* m)
        : Hittable(HittableType::BOX), num_sides(6), mat(m) {

        Point3 min_p(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
        Point3 max_p(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));

        Vec3 dx(max_p.x - min_p.x, 0, 0);
        Vec3 dy(0, max_p.y - min_p.y, 0);
        Vec3 dz(0, 0, max_p.z - min_p.z);

        // Front
        sides[0] = Quad(Point3(min_p.x, min_p.y, max_p.z), dx, dy, m);
        // Right
        sides[1] = Quad(Point3(max_p.x, min_p.y, max_p.z), -dz, dy, m);
        // Back
        sides[2] = Quad(Point3(max_p.x, min_p.y, min_p.z), -dx, dy, m);
        // Left
        sides[3] = Quad(Point3(min_p.x, min_p.y, min_p.z), dz, dy, m);
        // Top
        sides[4] = Quad(Point3(min_p.x, max_p.y, max_p.z), dx, -dz, m);
        // Bottom
        sides[5] = Quad(Point3(min_p.x, min_p.y, min_p.z), dx, dz, m);

        bbox = AABB(min_p, max_p);
    }

    __host__ __device__ bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const {
        HitRecord temp_rec;
        bool hit_anything = false;
        float closest_so_far = ray_t.max;

        for (int i = 0; i < num_sides; i++) {
            if (sides[i].hit(r, Interval(ray_t.min, closest_so_far), temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }
};

} // namespace rt

#endif // RAYTRACER_GEOMETRY_BOX_CUH
