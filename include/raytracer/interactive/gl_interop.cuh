#ifndef RAYTRACER_INTERACTIVE_GL_INTEROP_CUH
#define RAYTRACER_INTERACTIVE_GL_INTEROP_CUH

#ifdef ENABLE_INTERACTIVE

/**
 * @file gl_interop.cuh
 * @brief Interoperabilite CUDA-OpenGL pour l'affichage en temps reel
 * @details Ce fichier contient le kernel de conversion HDR vers RGBA 8-bit ainsi que
 *          la classe GLInterop qui gere le partage de memoire entre CUDA et OpenGL
 *          via un Pixel Buffer Object (PBO). Cela permet d'afficher directement le
 *          resultat du raytracing GPU dans une fenetre OpenGL sans copie CPU intermediaire.
 */

#include <GL/glew.h>
#include <GL/gl.h>
#include <cuda_gl_interop.h>
#include "raytracer/core/cuda_utils.cuh"
#include "raytracer/core/vec3.cuh"

namespace rt {

/**
 * @brief Kernel CUDA qui convertit un buffer HDR en RGBA 8-bit pour l'affichage
 * @details Ce kernel effectue trois operations sur chaque pixel :
 *          1. Moyenne des echantillons accumules (division par le nombre d'echantillons)
 *          2. Tone mapping Reinhard (c / (1 + c)) pour compresser les valeurs HDR en [0,1]
 *          3. Correction gamma (gamma = 2.2) pour un affichage correct sur ecran
 *          Le resultat est stocke au format uchar4 (RGBA, 8 bits par canal).
 * @param output Buffer de sortie RGBA 8-bit (un uchar4 par pixel)
 * @param hdr_buffer Buffer d'entree contenant les couleurs HDR accumulees
 * @param width Largeur de l'image en pixels
 * @param height Hauteur de l'image en pixels
 * @param accumulated_samples Nombre d'echantillons accumules dans le buffer HDR
 */
__global__ void convert_to_rgba8(
    uchar4* output,
    const Color* hdr_buffer,
    int width, int height,
    int accumulated_samples
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height) return;

    int idx = j * width + i;

    Color c = hdr_buffer[idx];
    if (accumulated_samples > 0) {
        c = c / static_cast<float>(accumulated_samples);
    }

    c = Color(
        c.x / (1.0f + c.x),
        c.y / (1.0f + c.y),
        c.z / (1.0f + c.z)
    );

    constexpr float inv_gamma = 1.0f / 2.2f;
    c = Color(
        powf(fmaxf(0.0f, c.x), inv_gamma),
        powf(fmaxf(0.0f, c.y), inv_gamma),
        powf(fmaxf(0.0f, c.z), inv_gamma)
    );

    output[idx] = make_uchar4(
        static_cast<unsigned char>(fminf(255.0f, c.x * 255.999f)),
        static_cast<unsigned char>(fminf(255.0f, c.y * 255.999f)),
        static_cast<unsigned char>(fminf(255.0f, c.z * 255.999f)),
        255
    );
}

/**
 * @class GLInterop
 * @brief Gere l'interoperabilite entre CUDA et OpenGL via un PBO
 * @details Cette classe cree et gere un Pixel Buffer Object (PBO) OpenGL qui est
 *          enregistre aupres de CUDA. Cela permet au kernel CUDA d'ecrire directement
 *          dans la memoire du PBO sans passer par le CPU. Le PBO est ensuite utilise
 *          pour mettre a jour une texture OpenGL qui est affichee a l'ecran.
 *          Le flux de donnees est : CUDA -> PBO -> Texture -> Affichage.
 */
class GLInterop {
public:
    GLuint pbo;                      ///< Identifiant du Pixel Buffer Object OpenGL
    GLuint texture;                  ///< Identifiant de la texture OpenGL pour l'affichage
    cudaGraphicsResource* cuda_pbo;  ///< Ressource CUDA associee au PBO pour l'interop
    int width, height;               ///< Dimensions de l'image en pixels

    /**
     * @brief Constructeur par defaut
     * @details Initialise tous les identifiants a zero/nullptr.
     */
    GLInterop() : pbo(0), texture(0), cuda_pbo(nullptr), width(0), height(0) {}

    /**
     * @brief Initialise le PBO, l'enregistre aupres de CUDA et cree la texture OpenGL
     * @details Cette methode :
     *          1. Cree un PBO OpenGL de taille width * height * sizeof(uchar4)
     *          2. Enregistre le PBO aupres de CUDA avec le flag WriteDiscard
     *          3. Cree une texture OpenGL RGBA8 avec filtrage lineaire
     * @param w Largeur de l'image en pixels
     * @param h Hauteur de l'image en pixels
     * @return true si l'initialisation a reussi, false en cas d'erreur CUDA
     */
    bool initialize(int w, int h) {
        width = w;
        height = h;

        glGenBuffers(1, &pbo);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBufferData(GL_PIXEL_UNPACK_BUFFER,
                     width * height * sizeof(uchar4),
                     nullptr,
                     GL_DYNAMIC_DRAW);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        cudaError_t err = cudaGraphicsGLRegisterBuffer(
            &cuda_pbo, pbo,
            cudaGraphicsMapFlagsWriteDiscard
        );
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to register PBO with CUDA: %s\n",
                    cudaGetErrorString(err));
            return false;
        }

        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,
                     width, height, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);

        return true;
    }

    /**
     * @brief Mappe le PBO dans l'espace memoire CUDA pour ecriture
     * @details Appelle cudaGraphicsMapResources pour rendre le PBO accessible
     *          depuis un kernel CUDA. Le pointeur retourne peut etre passe
     *          directement a un kernel pour y ecrire les donnees de l'image.
     * @return Pointeur device vers le buffer uchar4 du PBO
     */
    uchar4* map_for_cuda() {
        cudaGraphicsMapResources(1, &cuda_pbo, 0);

        uchar4* d_ptr;
        size_t size;
        cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&d_ptr), &size, cuda_pbo);

        return d_ptr;
    }

    /**
     * @brief Libere le PBO de l'acces CUDA
     * @details Doit etre appele apres que le kernel CUDA a fini d'ecrire dans le PBO.
     *          Apres cet appel, OpenGL peut a nouveau acceder au PBO.
     */
    void unmap_from_cuda() {
        cudaGraphicsUnmapResources(1, &cuda_pbo, 0);
    }

    /**
     * @brief Affiche le contenu du PBO a l'ecran via une texture OpenGL
     * @details Transfere les donnees du PBO vers la texture, puis dessine un quad
     *          plein ecran texture avec les coordonnees [-1,1] en projection orthographique.
     *          Les coordonnees de texture sont inversees en Y pour corriger l'orientation.
     */
    void display() {
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                       width, height,
                       GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        glEnable(GL_TEXTURE_2D);
        glBegin(GL_QUADS);
            glTexCoord2f(0, 1); glVertex2f(-1, -1);
            glTexCoord2f(1, 1); glVertex2f( 1, -1);
            glTexCoord2f(1, 0); glVertex2f( 1,  1);
            glTexCoord2f(0, 0); glVertex2f(-1,  1);
        glEnd();
        glDisable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    /**
     * @brief Libere toutes les ressources CUDA et OpenGL
     * @details Desenregistre la ressource CUDA, puis supprime le PBO et la texture
     *          OpenGL. Doit etre appele avant la destruction de l'objet.
     */
    void cleanup() {
        if (cuda_pbo) {
            cudaGraphicsUnregisterResource(cuda_pbo);
            cuda_pbo = nullptr;
        }
        if (pbo) {
            glDeleteBuffers(1, &pbo);
            pbo = 0;
        }
        if (texture) {
            glDeleteTextures(1, &texture);
            texture = 0;
        }
    }

    /**
     * @brief Destructeur
     * @details Ne libere pas automatiquement les ressources. Il faut appeler
     *          cleanup() explicitement avant la destruction.
     */
    ~GLInterop() {}
};

}

#endif
#endif
