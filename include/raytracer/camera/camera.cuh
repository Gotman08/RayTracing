#ifndef RAYTRACER_CAMERA_CAMERA_CUH
#define RAYTRACER_CAMERA_CAMERA_CUH

/** @file camera.cuh
 *  @brief Camera perspective avec DoF et motion blur */

#include "raytracer/core/vec3.cuh"
#include "raytracer/core/ray.cuh"
#include "raytracer/core/cuda_utils.cuh"
#include "raytracer/core/random.cuh"

namespace rt {

/** @brief Camera virtuelle : projection, DoF, motion blur */
class Camera {
public:
    Point3 center;
    Point3 pixel00_loc;
    Vec3 pixel_delta_u;
    Vec3 pixel_delta_v;
    Vec3 u, v, w;           ///< base ortho (u=droite, v=haut, w=arriere)
    float defocus_angle;     ///< 0 = pas de DoF
    Vec3 defocus_disk_u;
    Vec3 defocus_disk_v;
    float shutter_open;
    float shutter_close;

    int image_width;
    int image_height;

    /** @brief Init viewport, base ortho, defocus - a appeler avant get_ray */
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

        float theta = degrees_to_radians(vfov);
        float h = tanf(theta / 2.0f);
        float aspect_ratio = static_cast<float>(width) / static_cast<float>(height);
        float viewport_height = 2.0f * h * focus_dist;
        float viewport_width = viewport_height * aspect_ratio;

        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        Vec3 viewport_u = viewport_width * u;
        Vec3 viewport_v = viewport_height * (-v);

        pixel_delta_u = viewport_u / static_cast<float>(width);
        pixel_delta_v = viewport_v / static_cast<float>(height);

        Point3 viewport_upper_left = center - (focus_dist * w) - viewport_u / 2.0f - viewport_v / 2.0f;
        pixel00_loc = viewport_upper_left + 0.5f * (pixel_delta_u + pixel_delta_v);

        defocus_angle = aperture;
        float defocus_radius = focus_dist * tanf(degrees_to_radians(defocus_angle / 2.0f));
        defocus_disk_u = u * defocus_radius;
        defocus_disk_v = v * defocus_radius;
    }

    /** @brief Genere un rayon GPU pour pixel (i,j) avec jitter + DoF */
    __device__ Ray get_ray(int i, int j, curandState* rand_state) const {
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

    /** @brief Version CPU de get_ray (meme logique, rng CPU) */
    Ray get_ray_cpu(int i, int j, CPURandom& rng) const {
        float px = rng() - 0.5f;
        float py = rng() - 0.5f;

        Point3 pixel_sample = pixel00_loc
            + ((static_cast<float>(i) + px) * pixel_delta_u)
            + ((static_cast<float>(j) + py) * pixel_delta_v);

        Point3 ray_origin = (defocus_angle <= 0) ? center : defocus_disk_sample_cpu(rng);
        Vec3 ray_direction = pixel_sample - ray_origin;

        float ray_time = shutter_open + rng() * (shutter_close - shutter_open);

        return Ray(ray_origin, ray_direction, ray_time);
    }

private:
    /** @brief Sample sur le disque de defocus (GPU) */
    __device__ Point3 defocus_disk_sample(curandState* rand_state) const {
        Vec3 p = random_in_unit_disk(rand_state);
        return center + (p.x * defocus_disk_u) + (p.y * defocus_disk_v);
    }

    /** @brief Sample sur le disque de defocus (CPU) */
    Point3 defocus_disk_sample_cpu(CPURandom& rng) const {
        Vec3 p = random_in_unit_disk(rng);
        return center + (p.x * defocus_disk_u) + (p.y * defocus_disk_v);
    }
};

}

#endif
