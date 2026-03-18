#ifndef RAYTRACER_CORE_CUDA_COMPAT_CUH
#define RAYTRACER_CORE_CUDA_COMPAT_CUH

/**
 * @file cuda_compat.cuh
 * @brief Couche de compatibilite pour compiler le code sans le toolkit CUDA
 * @details Ce fichier permet de compiler le meme code source aussi bien
 *          avec le compilateur CUDA (nvcc) qu'avec un compilateur C++ standard.
 *          Lorsque __CUDACC__ n'est pas defini (compilation sans CUDA), les macros
 *          specifiques a CUDA (__host__, __device__, __global__, __constant__)
 *          sont definies comme vides, et une declaration forward de curandState
 *          est fournie. Cela permet au code de rester syntaxiquement valide
 *          meme sans le runtime CUDA, ce qui est utile pour le rendu CPU
 *          et pour les tests unitaires.
 */

#ifdef __CUDACC__
    #include <cuda_runtime.h>
#else
    #ifndef __host__
        #define __host__       ///< Macro vide en mode CPU (remplace le qualificateur CUDA __host__)
    #endif

    #ifndef __device__
        #define __device__     ///< Macro vide en mode CPU (remplace le qualificateur CUDA __device__)
    #endif

    #ifndef __global__
        #define __global__     ///< Macro vide en mode CPU (remplace le qualificateur CUDA __global__)
    #endif

    #ifndef __constant__
        #define __constant__   ///< Macro vide en mode CPU (remplace le qualificateur CUDA __constant__)
    #endif

    /**
     * @brief Declaration forward de curandState pour la compilation sans CUDA
     * @details Permet aux headers qui referencent curandState de compiler
     *          meme sans le toolkit CUDA installe. La structure n'est jamais
     *          instanciee en mode CPU.
     */
    struct curandState;

#endif

#endif
