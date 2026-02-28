#ifndef RAYTRACER_CAMERA_CAMERA_CUH
#define RAYTRACER_CAMERA_CAMERA_CUH

#include "raytracer/core/vec3.cuh"
#include "raytracer/core/ray.cuh"
#include "raytracer/core/cuda_utils.cuh"

namespace rt {

class Camera {
public:
    Point3 center;
    Point3 pixel00_loc;
    Vec3 pixel_delta_u;
    Vec3 pixel_delta_v;
    Vec3 u, v, w;
    float defocus_angle;
    Vec3 defocus_disk_u;
    Vec3 defocus_disk_v;
    float shutter_open;
    float shutter_close;

    int image_width;
    int image_height;

    __host__ void initialize(
        int width, int height,
        Point3 lookfrom, Point3 lookat, Vec3 vup,
        float vfov, float aperture = 0.0f, float focus_dist = 10.0f,
        float t0 = 0.0f, float t1 = 0.0f
    ) {
        image_width = width;
        image_height = height;

        center = lookfrom;
        shutter_open = t0;
        shutter_close = t1;

        // Viewport dimensions
        float theta = degrees_to_radians(vfov);
        float h = tanf(theta / 2.0f);
        float aspect_ratio = static_cast<float>(width) / static_cast<float>(height);
        float viewport_height = 2.0f * h * focus_dist;
        float viewport_width = viewport_height * aspect_ratio;

        // Camera coordinate system
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        // Viewport edge vectors
        Vec3 viewport_u = viewport_width * u;
        Vec3 viewport_v = viewport_height * (-v);

        // Pixel delta vectors
        pixel_delta_u = viewport_u / static_cast<float>(width);
        pixel_delta_v = viewport_v / static_cast<float>(height);

        // Upper left pixel location
        Point3 viewport_upper_left = center - (focus_dist * w) - viewport_u / 2.0f - viewport_v / 2.0f;
        pixel00_loc = viewport_upper_left + 0.5f * (pixel_delta_u + pixel_delta_v);

        // Defocus disk
        defocus_angle = aperture;
        float defocus_radius = focus_dist * tanf(degrees_to_radians(defocus_angle / 2.0f));
        defocus_disk_u = u * defocus_radius;
        defocus_disk_v = v * defocus_radius;
    }

    __device__ Ray get_ray(int i, int j, curandState* rand_state) const {
        // Random offset within pixel
        float px = curand_uniform(rand_state) - 0.5f;
        float py = curand_uniform(rand_state) - 0.5f;

        Point3 pixel_sample = pixel00_loc
            + ((static_cast<float>(i) + px) * pixel_delta_u)
            + ((static_cast<float>(j) + py) * pixel_delta_v);

        Point3 ray_origin = (defocus_angle <= 0) ? center : defocus_disk_sample(rand_state);
        Vec3 ray_direction = pixel_sample - ray_origin;

        float ray_time = shutter_open + curand_uniform(rand_state) * (shutter_close - shutter_open);

        return Ray(ray_origin, ray_direction, ray_time);
    }

private:
    __device__ Point3 defocus_disk_sample(curandState* rand_state) const {
        Vec3 p = random_in_unit_disk(rand_state);
        return center + (p.x * defocus_disk_u) + (p.y * defocus_disk_v);
    }
};

} // namespace rt

#endif // RAYTRACER_CAMERA_CAMERA_CUH
