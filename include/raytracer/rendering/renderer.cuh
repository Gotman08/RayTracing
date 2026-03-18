#ifndef RAYTRACER_RENDERING_RENDERER_CUH
#define RAYTRACER_RENDERING_RENDERER_CUH

/**
 * @file renderer.cuh
 * @brief Moteur de rendu GPU principal du raytracer
 * @details Ce fichier contient les kernels CUDA et fonctions device responsables
 *          du rendu par lancer de rayons sur GPU. On y trouve le calcul de la couleur
 *          d'un rayon par rebonds successifs, le kernel principal de rendu avec
 *          multi-echantillonnage, ainsi que les kernels d'initialisation des etats
 *          aleatoires et le kernel de rendu progressif pour le mode interactif.
 */

#include "raytracer/core/cuda_utils.cuh"
#include "raytracer/core/vec3.cuh"
#include "raytracer/core/hit_record.cuh"
#include "raytracer/camera/camera.cuh"
#include "raytracer/geometry/hittable_list.cuh"
#include "raytracer/acceleration/bvh.cuh"
#include "raytracer/rendering/render_config.cuh"
#include "raytracer/rendering/material_dispatch.cuh"
#include "raytracer/rendering/tone_mapping.cuh"

namespace rt {

/**
 * @brief Calcule la couleur d'un rayon en simulant ses rebonds successifs dans la scene
 * @details Cette fonction utilise une boucle iterative (et non recursive) pour tracer
 *          le parcours d'un rayon a travers la scene. A chaque rebond, on accumule
 *          l'attenuation du materiau touche. Si le rayon ne touche aucun objet,
 *          on retourne la couleur du ciel (ou du fond) ponderee par l'attenuation
 *          accumulee. Si le rayon depasse la profondeur maximale, la contribution
 *          est nulle (noir).
 * @param initial_ray Le rayon initial lance depuis la camera
 * @param bvh La structure d'acceleration BVH contenant la scene
 * @param max_depth Le nombre maximal de rebonds autorises
 * @param config La configuration du rendu (ciel, fond, etc.)
 * @param rand_state L'etat du generateur aleatoire curand pour ce thread
 * @return La couleur finale calculee pour ce rayon
 */
__device__ inline Color ray_color(
    const Ray& initial_ray,
    const BVH& bvh,
    int max_depth,
    const RenderConfig& config,
    curandState* rand_state
) {
    Color accumulated(0, 0, 0);
    Color attenuation(1, 1, 1);
    Ray current_ray = initial_ray;

    for (int depth = 0; depth < max_depth; depth++) {
        HitRecord rec;

        if (bvh.hit(current_ray, Interval(0.001f, INFINITY_F), rec)) {
            Ray scattered;
            Color scatter_attenuation;

            if (scatter(*rec.mat, current_ray, rec, scatter_attenuation, scattered, rand_state)) {
                attenuation = attenuation * scatter_attenuation;
                current_ray = scattered;
            } else {
                break;
            }
        } else {
            if (config.use_sky) {
                accumulated += attenuation * config.sky.get_color(current_ray.direction());
            } else {
                accumulated += attenuation * config.background;
            }
            break;
        }
    }

    return accumulated;
}

/**
 * @brief Kernel principal de rendu GPU : chaque thread calcule la couleur d'un pixel
 * @details Chaque thread CUDA correspond a un pixel de l'image. Pour chaque pixel,
 *          on lance plusieurs rayons (multi-echantillonnage / antialiasing) a travers
 *          la scene, puis on fait la moyenne des couleurs obtenues. Ensuite, on applique
 *          le tone mapping de Reinhard et la correction gamma avant de clamper les
 *          valeurs dans [0, 1] et d'ecrire le resultat dans le frame buffer.
 * @param frame_buffer Le buffer de sortie contenant les couleurs des pixels
 * @param camera La camera utilisee pour generer les rayons
 * @param bvh La structure d'acceleration BVH de la scene
 * @param config La configuration du rendu (resolution, samples, profondeur, etc.)
 * @param rand_states Les etats aleatoires curand, un par pixel
 */
__global__ void render_kernel(
    Color* frame_buffer,
    Camera camera,
    BVH bvh,
    RenderConfig config,
    curandState* rand_states
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= config.width || j >= config.height) return;

    int pixel_index = j * config.width + i;
    curandState* local_rand_state = &rand_states[pixel_index];

    Color pixel_color(0, 0, 0);

    for (int s = 0; s < config.samples_per_pixel; s++) {
        Ray r = camera.get_ray(i, j, local_rand_state);
        pixel_color += ray_color(r, bvh, config.max_depth, config, local_rand_state);
    }

    pixel_color = pixel_color / static_cast<float>(config.samples_per_pixel);

    pixel_color = apply_tone_mapping(pixel_color);
    pixel_color = gamma_correct(pixel_color);

    pixel_color.x = fminf(1.0f, fmaxf(0.0f, pixel_color.x));
    pixel_color.y = fminf(1.0f, fmaxf(0.0f, pixel_color.y));
    pixel_color.z = fminf(1.0f, fmaxf(0.0f, pixel_color.z));

    frame_buffer[pixel_index] = pixel_color;
}

/**
 * @brief Fonction de hachage rapide de Wang pour generer des graines pseudo-aleatoires
 * @details Cette fonction prend un entier en entree et produit un hachage bien distribue.
 *          Elle est utilisee pour creer des graines uniques et variees pour chaque pixel
 *          lors de l'initialisation des etats curand, ce qui evite les correlations
 *          entre pixels voisins. C'est beaucoup plus rapide que curand_init avec
 *          des sequences differentes.
 * @param seed La valeur d'entree a hacher (typiquement l'index du pixel + un offset)
 * @return La valeur hachee, bien distribuee sur 32 bits
 */
__device__ __forceinline__ unsigned int wang_hash(unsigned int seed) {
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

/**
 * @brief Kernel d'initialisation rapide des etats aleatoires curand via hachage
 * @details Chaque thread initialise l'etat curand de son pixel en utilisant le
 *          hachage de Wang pour creer une graine unique. Cette methode est bien plus
 *          rapide que l'initialisation classique avec des sequences differentes,
 *          car curand_init est couteux quand la sequence est grande.
 * @param rand_states Le tableau des etats curand a initialiser (un par pixel)
 * @param width La largeur de l'image en pixels
 * @param height La hauteur de l'image en pixels
 * @param seed La graine de base pour le hachage
 */
__global__ void init_rand_states_fast(
    curandState* rand_states,
    int width, int height,
    unsigned int seed
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height) return;

    int pixel_index = j * width + i;
    unsigned int hash = wang_hash(seed + pixel_index);
    curand_init(hash, 0, 0, &rand_states[pixel_index]);
}

/**
 * @brief Kernel d'initialisation classique des etats aleatoires curand
 * @details Chaque thread initialise son etat curand en utilisant une graine basee
 *          sur l'index du pixel. Cette methode est plus lente que la version rapide
 *          (init_rand_states_fast) car curand_init effectue un travail interne plus
 *          important, mais elle garantit des sequences parfaitement independantes.
 * @param rand_states Le tableau des etats curand a initialiser (un par pixel)
 * @param width La largeur de l'image en pixels
 * @param height La hauteur de l'image en pixels
 * @param seed La graine de base pour l'initialisation
 */
__global__ void init_rand_states(
    curandState* rand_states,
    int width, int height,
    unsigned long long seed
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height) return;

    int pixel_index = j * width + i;
    curand_init(seed + pixel_index, 0, 0, &rand_states[pixel_index]);
}

#ifdef ENABLE_INTERACTIVE
/**
 * @brief Kernel de rendu progressif pour le mode interactif
 * @details Ce kernel est utilise en mode interactif (temps reel). Au lieu de calculer
 *          tous les samples d'un coup, il accumule progressivement les echantillons
 *          dans un buffer d'accumulation. Le tone mapping et la correction gamma ne
 *          sont pas appliques ici : ils seront appliques plus tard lors de l'affichage,
 *          ce qui permet de continuer a accumuler des samples au fil des frames.
 * @param accumulation_buffer Le buffer d'accumulation des couleurs (somme des samples)
 * @param camera La camera utilisee pour generer les rayons
 * @param bvh La structure d'acceleration BVH de la scene
 * @param config La configuration du rendu
 * @param rand_states Les etats aleatoires curand, un par pixel
 * @param samples_this_frame Le nombre de samples a calculer pour cette frame
 */
__global__ void render_kernel_accumulate(
    Color* accumulation_buffer,
    Camera camera,
    BVH bvh,
    RenderConfig config,
    curandState* rand_states,
    int samples_this_frame
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= config.width || j >= config.height) return;

    int pixel_index = j * config.width + i;
    curandState* local_rand_state = &rand_states[pixel_index];

    Color pixel_color(0, 0, 0);

    for (int s = 0; s < samples_this_frame; s++) {
        Ray r = camera.get_ray(i, j, local_rand_state);
        pixel_color += ray_color(r, bvh, config.max_depth, config, local_rand_state);
    }

    accumulation_buffer[pixel_index] = accumulation_buffer[pixel_index] + pixel_color;
}
#endif

}

#endif
