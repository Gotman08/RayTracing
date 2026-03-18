#ifndef RAYTRACER_INTERACTIVE_ACCUMULATION_BUFFER_CUH
#define RAYTRACER_INTERACTIVE_ACCUMULATION_BUFFER_CUH

#ifdef ENABLE_INTERACTIVE

/**
 * @file accumulation_buffer.cuh
 * @brief Buffer d'accumulation progressive pour le rendu interactif
 * @details Ce fichier contient les kernels CUDA et la classe pour gerer l'accumulation
 *          progressive des echantillons de raytracing. L'idee est d'additionner les
 *          resultats de plusieurs passes de rendu pour reduire le bruit progressivement.
 *          Quand la camera bouge, le buffer est remis a zero pour recommencer l'accumulation.
 */

#include "raytracer/core/cuda_utils.cuh"
#include "raytracer/core/vec3.cuh"

namespace rt {

/**
 * @brief Kernel CUDA qui remet a zero le buffer d'accumulation
 * @details Chaque thread met un pixel du buffer a la couleur noire (0, 0, 0).
 *          La grille 2D de threads couvre toute l'image. Les threads hors limites
 *          sont ignores grace a la condition de bord.
 * @param buffer Pointeur device vers le buffer d'accumulation a reinitialiser
 * @param width Largeur de l'image en pixels
 * @param height Hauteur de l'image en pixels
 */
__global__ void clear_accumulation_kernel(Color* buffer, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height) return;

    int idx = j * width + i;
    buffer[idx] = Color(0, 0, 0);
}

/**
 * @brief Kernel CUDA qui ajoute de nouveaux echantillons au buffer d'accumulation
 * @details Pour chaque pixel, additionne la couleur des nouveaux echantillons a la
 *          valeur deja accumulee. Cela permet de faire converger l'image progressivement
 *          en additionnant de plus en plus d'echantillons.
 * @param accumulation_buffer Buffer d'accumulation (lecture/ecriture) contenant la somme des echantillons
 * @param new_samples Buffer en lecture seule contenant les nouveaux echantillons a ajouter
 * @param width Largeur de l'image en pixels
 * @param height Hauteur de l'image en pixels
 */
__global__ void accumulate_samples_kernel(
    Color* accumulation_buffer,
    const Color* new_samples,
    int width, int height
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height) return;

    int idx = j * width + i;
    accumulation_buffer[idx] = accumulation_buffer[idx] + new_samples[idx];
}

/**
 * @class AccumulationBuffer
 * @brief Buffer d'accumulation progressive pour le rendu interactif
 * @details Cette classe gere un buffer GPU qui accumule les resultats de plusieurs
 *          passes de rendu. A chaque frame, les nouveaux echantillons sont additionnes
 *          au buffer, et le nombre total d'echantillons est incremente. Cela permet
 *          d'obtenir une image de plus en plus propre quand la camera est immobile.
 *          Quand la camera bouge, on appelle reset() pour repartir de zero.
 */
class AccumulationBuffer {
public:
    Color* d_buffer;        ///< Buffer GPU contenant la somme des echantillons accumules
    int width, height;      ///< Dimensions de l'image en pixels
    int accumulated_samples; ///< Nombre total d'echantillons accumules jusqu'ici

    /**
     * @brief Constructeur par defaut
     * @details Initialise le buffer a nullptr et les compteurs a zero.
     */
    AccumulationBuffer()
        : d_buffer(nullptr), width(0), height(0), accumulated_samples(0) {}

    /**
     * @brief Alloue le buffer GPU et le remet a zero
     * @details Alloue un tableau de Color (float3) sur le GPU de taille width * height,
     *          puis appelle reset() pour initialiser tous les pixels a noir.
     * @param w Largeur de l'image en pixels
     * @param h Hauteur de l'image en pixels
     * @return true si l'allocation a reussi, false en cas d'erreur CUDA
     */
    bool initialize(int w, int h) {
        width = w;
        height = h;
        accumulated_samples = 0;

        cudaError_t err = cudaMalloc(&d_buffer, width * height * sizeof(Color));
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to allocate accumulation buffer: %s\n",
                    cudaGetErrorString(err));
            return false;
        }

        reset();
        return true;
    }

    /**
     * @brief Remet le buffer a zero et reinitialise le compteur d'echantillons
     * @details Lance le kernel clear_accumulation_kernel pour mettre tous les pixels
     *          a noir (0,0,0), puis remet le compteur d'echantillons a zero.
     *          Appele quand la camera change de position ou d'orientation.
     */
    void reset() {
        accumulated_samples = 0;

        constexpr int BLOCK_SIZE = 16;
        dim3 blocks((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                    (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

        clear_accumulation_kernel<<<blocks, threads>>>(d_buffer, width, height);
        cudaDeviceSynchronize();
    }

    /**
     * @brief Ajoute de nouveaux echantillons au buffer d'accumulation
     * @details Lance le kernel accumulate_samples_kernel pour additionner les nouveaux
     *          echantillons pixel par pixel, puis met a jour le compteur total.
     * @param d_new_samples Pointeur device vers le buffer des nouveaux echantillons
     * @param new_sample_count Nombre d'echantillons par pixel dans cette passe
     */
    void accumulate(const Color* d_new_samples, int new_sample_count) {
        constexpr int BLOCK_SIZE = 16;
        dim3 blocks((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                    (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

        accumulate_samples_kernel<<<blocks, threads>>>(
            d_buffer, d_new_samples, width, height);

        accumulated_samples += new_sample_count;
    }

    /**
     * @brief Retourne le nombre total d'echantillons accumules
     * @return Nombre d'echantillons par pixel accumules depuis le dernier reset
     */
    int get_accumulated_samples() const {
        return accumulated_samples;
    }

    /**
     * @brief Libere la memoire GPU du buffer
     * @details Doit etre appele avant la destruction de l'objet pour eviter
     *          les fuites de memoire GPU.
     */
    void cleanup() {
        if (d_buffer) {
            cudaFree(d_buffer);
            d_buffer = nullptr;
        }
    }

    /**
     * @brief Destructeur
     * @details Ne libere pas automatiquement la memoire GPU. Il faut appeler
     *          cleanup() explicitement avant la destruction.
     */
    ~AccumulationBuffer() {}
};

}

#endif
#endif
