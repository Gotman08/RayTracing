#ifndef RAYTRACER_RENDERING_RENDERER_CUH
#define RAYTRACER_RENDERING_RENDERER_CUH

/** @file renderer.cuh
 * @brief Kernels CUDA du moteur de rendu GPU (ray tracing + tone mapping) */

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

// -----------------------------------------------
// Memoire constante : Camera et RenderConfig sont lues par tous les threads
// de maniere identique → le cache constant broadcast la valeur a tout le warp
// en un seul cycle (~5 cycles vs ~500 en memoire globale).
// Taille totale : ~184 octets, bien en dessous de la limite de 64 KB.
// Stockage brut aligne pour eviter l'erreur "dynamic initialization is not
// supported for a __constant__ variable" (CUDA >= 12).
// -----------------------------------------------
__constant__ char d_const_camera_buf[sizeof(Camera)];
__constant__ char d_const_config_buf[sizeof(RenderConfig)];
#define d_const_camera  (*(const Camera*)d_const_camera_buf)
#define d_const_config  (*(const RenderConfig*)d_const_config_buf)

/** @brief Trace un rayon dans la scene par rebonds iteratifs, retourne la couleur accumulee */
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

/** @brief Kernel principal : 1 thread = 1 pixel, multi-sampling puis stockage HDR brut */
__global__ void render_kernel(
    Color* frame_buffer,
    BVH bvh,
    curandState* rand_states
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= d_const_config.width || j >= d_const_config.height) return;

    int pixel_index = j * d_const_config.width + i;
    curandState* local_rand_state = &rand_states[pixel_index];

    Color pixel_color(0, 0, 0);

    for (int s = 0; s < d_const_config.samples_per_pixel; s++) {
        Ray r = d_const_camera.get_ray(i, j, local_rand_state);
        pixel_color += ray_color(r, bvh, d_const_config.max_depth, d_const_config, local_rand_state);
    }

    // moyenne des samples, on garde le HDR brut pour le tone mapping adaptatif
    pixel_color = pixel_color / static_cast<float>(d_const_config.samples_per_pixel);

    frame_buffer[pixel_index] = pixel_color;
}

/** @brief Hash de Wang - graine pseudo-aleatoire rapide pour init curand */
__device__ __forceinline__ unsigned int wang_hash(unsigned int seed) {
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

/** @brief Init rapide des etats curand via wang_hash (1 thread/pixel) */
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

/** @brief Init classique des etats curand (plus lent, sequences independantes) */
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

// -----------------------------------------------
// Reduction parallele : calcul de la luminance moyenne de l'image
// Utilise __shared__ memory pour la reduction intra-bloc (arbre binaire)
// puis atomicAdd pour combiner les resultats des blocs.
// Necessaire pour le tone mapping adaptatif de Reinhard etendu.
// -----------------------------------------------

/** @brief Reduction parallele : somme des luminances (shared mem + warp shuffle) */
__global__ void compute_avg_luminance(
    const Color* frame_buffer,
    float* d_total_luminance,
    int num_pixels
) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // charger la luminance du pixel dans la shared memory
    if (gid < num_pixels) {
        Color c = frame_buffer[gid];
        sdata[tid] = luminance(c);
    } else {
        sdata[tid] = 0.0f;
    }
    __syncthreads();

    // Etape 1 : reduction arbre binaire en shared memory pour stride > 32
    // (stride=128 puis stride=64 : 2 iterations, 2 __syncthreads)
    for (unsigned int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Etape 2 : warp-level reduction pour les 32 derniers elements
    // Les threads 0-31 forment le warp 0, pas de __syncthreads necessaire.
    // __shfl_down_sync echange les valeurs de registres entre threads du meme warp.
    if (tid < 32) {
        float val = sdata[tid] + sdata[tid + 32];  // stride=32 directement en registre
        val += __shfl_down_sync(0xffffffff, val, 16);
        val += __shfl_down_sync(0xffffffff, val, 8);
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);
        if (tid == 0) atomicAdd(d_total_luminance, val);
    }
}

/** @brief Post-process : Reinhard adaptatif + gamma sRGB + clamp, in-place */
__global__ void apply_tonemapping_kernel(
    Color* frame_buffer,
    float avg_luminance,
    int num_pixels
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_pixels) return;

    Color hdr = frame_buffer[gid];

    // tone mapping adaptatif de Reinhard etendu
    Color mapped = apply_adaptive_tone_mapping(hdr, avg_luminance);

    // correction gamma pour l'espace sRGB
    mapped = gamma_correct(mapped);

    // clamp final dans [0, 1]
    mapped.x = fminf(1.0f, fmaxf(0.0f, mapped.x));
    mapped.y = fminf(1.0f, fmaxf(0.0f, mapped.y));
    mapped.z = fminf(1.0f, fmaxf(0.0f, mapped.z));

    frame_buffer[gid] = mapped;
}

}

#endif
