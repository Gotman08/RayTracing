#ifndef RAYTRACER_CORE_CUDA_UTILS_CUH
#define RAYTRACER_CORE_CUDA_UTILS_CUH

/**
 * @file cuda_utils.cuh
 * @brief Utilitaires CUDA pour le raytracer
 * @details Ce fichier regroupe les constantes mathematiques, les fonctions
 *          d'aide a la gestion de la memoire GPU (allocation, copie, liberation),
 *          la verification des erreurs CUDA, l'affichage des informations du GPU
 *          et l'initialisation des etats curand pour la generation de nombres
 *          aleatoires sur le GPU.
 */

#include "raytracer/core/cuda_compat.cuh"
#include <cstdio>
#include <cstdlib>

#ifdef __CUDACC__
#include <curand_kernel.h>
#endif

namespace rt {

constexpr float PI = 3.14159265358979323846f;  ///< Constante PI en simple precision
constexpr float INFINITY_F = 1e30f;            ///< Valeur representant l'infini pour les calculs de rayons
constexpr float EPSILON = 1e-6f;               ///< Seuil de tolerance pour les comparaisons flottantes

/**
 * @brief Convertit un angle en degres vers des radians
 * @details Fonction utilisable a la fois sur le CPU et le GPU grace aux
 *          qualificateurs __host__ __device__. Utile pour convertir les
 *          parametres de la camera (champ de vision) en radians.
 * @param degrees L'angle en degres a convertir
 * @return L'angle equivalent en radians
 */
__host__ __device__ inline float degrees_to_radians(float degrees) {
    return degrees * PI / 180.0f;
}

#ifdef __CUDACC__

/**
 * @brief Macro de verification des erreurs CUDA
 * @details Enveloppe un appel CUDA et verifie le code de retour.
 *          En cas d'erreur, affiche le fichier, la ligne et le message
 *          d'erreur CUDA, puis termine le programme.
 * @param val L'appel CUDA a verifier (ex : cudaMalloc(...))
 */
#define CUDA_CHECK(val) check_cuda((val), #val, __FILE__, __LINE__)

/**
 * @brief Verifie le resultat d'un appel CUDA et termine le programme en cas d'erreur
 * @details Fonction appelee par la macro CUDA_CHECK. Si le resultat n'est pas
 *          cudaSuccess, elle affiche un message d'erreur detaille avec le nom
 *          du fichier, le numero de ligne et la description de l'erreur,
 *          puis reinitialise le GPU et quitte.
 * @param result Le code d'erreur CUDA retourne par l'appel
 * @param func Le nom de la fonction CUDA sous forme de chaine (via #val)
 * @param file Le fichier source ou l'erreur s'est produite
 * @param line Le numero de ligne ou l'erreur s'est produite
 */
inline void check_cuda(cudaError_t result, const char* func,
                       const char* file, int line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Error at %s:%d - %s\n    %s: %s\n",
                file, line, func,
                cudaGetErrorName(result),
                cudaGetErrorString(result));
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())       ///< Verifie la derniere erreur CUDA survenue
#define CUDA_SYNC_CHECK() CUDA_CHECK(cudaDeviceSynchronize())  ///< Synchronise le GPU et verifie les erreurs

/**
 * @brief Alloue de la memoire sur le GPU (device)
 * @details Encapsule cudaMalloc avec verification automatique des erreurs.
 *          La memoire allouee n'est accessible que depuis le GPU.
 * @tparam T Le type des elements a allouer
 * @param count Le nombre d'elements a allouer
 * @return Pointeur vers la memoire GPU allouee
 */
template<typename T>
T* cuda_alloc(size_t count) {
    T* ptr;
    CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
    return ptr;
}

/**
 * @brief Alloue de la memoire unifiee (managed) accessible depuis le CPU et le GPU
 * @details Encapsule cudaMallocManaged avec verification des erreurs.
 *          La memoire unifiee est automatiquement migree entre CPU et GPU
 *          selon les besoins, ce qui simplifie le code mais peut impacter
 *          les performances.
 * @tparam T Le type des elements a allouer
 * @param count Le nombre d'elements a allouer
 * @return Pointeur vers la memoire unifiee allouee
 */
template<typename T>
T* cuda_alloc_managed(size_t count) {
    T* ptr;
    CUDA_CHECK(cudaMallocManaged(&ptr, count * sizeof(T)));
    return ptr;
}

/**
 * @brief Libere de la memoire GPU precedemment allouee
 * @details Encapsule cudaFree avec verification des erreurs.
 *          Verifie que le pointeur n'est pas nul avant de liberer.
 * @tparam T Le type des elements pointes
 * @param ptr Pointeur vers la memoire GPU a liberer
 */
template<typename T>
void cuda_free(T* ptr) {
    if (ptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
}

/**
 * @brief Copie des donnees du CPU (host) vers le GPU (device)
 * @details Encapsule cudaMemcpy en mode HostToDevice avec verification des erreurs.
 * @tparam T Le type des elements a copier
 * @param dst Pointeur destination sur le GPU
 * @param src Pointeur source sur le CPU
 * @param count Nombre d'elements a copier
 */
template<typename T>
void cuda_copy_to_device(T* dst, const T* src, size_t count) {
    CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyHostToDevice));
}

/**
 * @brief Copie des donnees du GPU (device) vers le CPU (host)
 * @details Encapsule cudaMemcpy en mode DeviceToHost avec verification des erreurs.
 * @tparam T Le type des elements a copier
 * @param dst Pointeur destination sur le CPU
 * @param src Pointeur source sur le GPU
 * @param count Nombre d'elements a copier
 */
template<typename T>
void cuda_copy_to_host(T* dst, const T* src, size_t count) {
    CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToHost));
}

/**
 * @brief Affiche les informations detaillees du ou des GPU disponibles
 * @details Interroge le driver CUDA pour obtenir et afficher les proprietes
 *          de chaque GPU : nom, capacite de calcul, memoire globale,
 *          nombre de multiprocesseurs, nombre max de threads, memoire partagee
 *          et taille de warp. Utile pour le diagnostic et le deboggage.
 */
inline void print_device_info() {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    printf("=== CUDA Device Information ===\n");
    printf("Number of CUDA devices: %d\n\n", device_count);

    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));

        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total global memory: %.2f GB\n",
               prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Shared memory per block: %.2f KB\n",
               prop.sharedMemPerBlock / 1024.0);
        printf("  Warp size: %d\n", prop.warpSize);
        printf("\n");
    }
}

/**
 * @brief Kernel GPU pour initialiser les etats du generateur curand
 * @details Chaque thread initialise l'etat curand correspondant a son pixel
 *          dans l'image. La graine est differente pour chaque pixel (seed + pixel_index)
 *          afin d'assurer l'independance des sequences aleatoires entre pixels.
 *          Les threads hors limites de l'image sont ignores.
 * @param states Tableau d'etats curand a initialiser (un par pixel)
 * @param width Largeur de l'image en pixels
 * @param height Hauteur de l'image en pixels
 * @param seed Graine de base pour l'initialisation des generateurs
 */
__global__ inline void init_curand_states(curandState* states, int width, int height, unsigned long long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height) return;

    int pixel_index = j * width + i;
    curand_init(seed + pixel_index, 0, 0, &states[pixel_index]);
}

#endif

}

#endif
