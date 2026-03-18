#ifndef RAYTRACER_CAMERA_CAMERA_CUH
#define RAYTRACER_CAMERA_CAMERA_CUH

/**
 * @file camera.cuh
 * @brief Definition de la classe Camera pour le lancer de rayons
 * @details Ce fichier contient la classe Camera qui gere la projection perspective,
 *          la profondeur de champ (depth of field) et le motion blur. La camera
 *          supporte a la fois le rendu GPU (CUDA) et le rendu CPU.
 */

#include "raytracer/core/vec3.cuh"
#include "raytracer/core/ray.cuh"
#include "raytracer/core/cuda_utils.cuh"
#include "raytracer/core/random.cuh"

namespace rt {

/**
 * @brief Camera avec projection perspective, profondeur de champ et motion blur
 * @details Cette classe represente une camera virtuelle dans la scene 3D. Elle permet
 *          de configurer la position, l'orientation, le champ de vision (FOV), l'ouverture
 *          pour la profondeur de champ, et l'obturateur pour le motion blur. La camera
 *          genere des rayons primaires pour chaque pixel de l'image avec un jitter
 *          aleatoire pour l'anti-aliasing.
 */
class Camera {
public:
    Point3 center;           ///< Position de la camera dans la scene
    Point3 pixel00_loc;      ///< Position du pixel (0,0) dans l'espace monde
    Vec3 pixel_delta_u;      ///< Decalage horizontal entre deux pixels adjacents
    Vec3 pixel_delta_v;      ///< Decalage vertical entre deux pixels adjacents
    Vec3 u, v, w;            ///< Base orthonormee de la camera (u=droite, v=haut, w=arriere)
    float defocus_angle;     ///< Angle d'ouverture pour la profondeur de champ (0 = pas de flou)
    Vec3 defocus_disk_u;     ///< Vecteur horizontal du disque de defocalisation
    Vec3 defocus_disk_v;     ///< Vecteur vertical du disque de defocalisation
    float shutter_open;      ///< Instant d'ouverture de l'obturateur (pour le motion blur)
    float shutter_close;     ///< Instant de fermeture de l'obturateur (pour le motion blur)

    int image_width;         ///< Largeur de l'image en pixels
    int image_height;        ///< Hauteur de l'image en pixels

    /**
     * @brief Configure la camera avec tous ses parametres
     * @details Calcule le viewport, la base orthonormee et les parametres de
     *          defocalisation a partir des parametres fournis. Doit etre appelee
     *          avant de generer des rayons.
     * @param width Largeur de l'image en pixels
     * @param height Hauteur de l'image en pixels
     * @param lookfrom Position de la camera dans la scene
     * @param lookat Point vise par la camera
     * @param vup Vecteur "haut" du monde (pour definir l'orientation)
     * @param vfov Champ de vision vertical en degres
     * @param aperture Angle d'ouverture pour la profondeur de champ (0 = pas de flou)
     * @param focus_dist Distance de mise au point
     * @param t0 Instant d'ouverture de l'obturateur
     * @param t1 Instant de fermeture de l'obturateur
     */
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

    /**
     * @brief Genere un rayon pour le pixel (i, j) sur GPU
     * @details Le rayon est perturbe aleatoirement a l'interieur du pixel (jitter)
     *          pour l'anti-aliasing. Si la profondeur de champ est activee, l'origine
     *          du rayon est echantillonnee sur le disque de defocalisation. Le temps
     *          du rayon est aleatoire entre shutter_open et shutter_close pour le motion blur.
     * @param i Coordonnee horizontale du pixel (colonne)
     * @param j Coordonnee verticale du pixel (ligne)
     * @param rand_state Etat du generateur aleatoire CUDA pour ce thread
     * @return Un rayon partant de la camera vers le pixel (i, j) avec jitter
     */
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

    /**
     * @brief Genere un rayon pour le pixel (i, j) sur CPU
     * @details Version CPU de get_ray(). Meme logique que la version GPU mais
     *          utilise un generateur aleatoire CPU au lieu de curand.
     * @param i Coordonnee horizontale du pixel (colonne)
     * @param j Coordonnee verticale du pixel (ligne)
     * @param rng Reference vers le generateur aleatoire CPU
     * @return Un rayon partant de la camera vers le pixel (i, j) avec jitter
     */
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
    /**
     * @brief Echantillonne un point aleatoire sur le disque de defocalisation (GPU)
     * @details Genere un point aleatoire dans le disque d'ouverture de la camera
     *          pour simuler la profondeur de champ. Plus le defocus_angle est grand,
     *          plus le flou est prononce pour les objets hors du plan de mise au point.
     * @param rand_state Etat du generateur aleatoire CUDA
     * @return Un point sur le disque de defocalisation autour du centre de la camera
     */
    __device__ Point3 defocus_disk_sample(curandState* rand_state) const {
        Vec3 p = random_in_unit_disk(rand_state);
        return center + (p.x * defocus_disk_u) + (p.y * defocus_disk_v);
    }

    /**
     * @brief Echantillonne un point aleatoire sur le disque de defocalisation (CPU)
     * @details Version CPU de defocus_disk_sample(). Meme logique mais utilise
     *          le generateur aleatoire CPU.
     * @param rng Reference vers le generateur aleatoire CPU
     * @return Un point sur le disque de defocalisation autour du centre de la camera
     */
    Point3 defocus_disk_sample_cpu(CPURandom& rng) const {
        Vec3 p = random_in_unit_disk(rng);
        return center + (p.x * defocus_disk_u) + (p.y * defocus_disk_v);
    }
};

}

#endif
