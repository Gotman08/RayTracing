#ifndef RAYTRACER_INTERACTIVE_WINDOW_CUH
#define RAYTRACER_INTERACTIVE_WINDOW_CUH

#ifdef ENABLE_INTERACTIVE

/**
 * @file window.cuh
 * @brief Gestion de la fenetre GLFW pour le mode interactif du raytracer
 * @details Ce fichier contient la classe Window qui encapsule la creation et la gestion
 *          d'une fenetre GLFW avec un contexte OpenGL. La fenetre est utilisee pour
 *          afficher le rendu en temps reel lors du mode interactif.
 */

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cstdio>

namespace rt {

/**
 * @class Window
 * @brief Fenetre GLFW pour l'affichage du rendu interactif
 * @details Cette classe encapsule toute la logique de creation, configuration et destruction
 *          d'une fenetre GLFW. Elle initialise aussi GLEW pour les extensions OpenGL et
 *          configure une projection orthographique simple pour l'affichage d'une texture
 *          plein ecran. Le curseur est capture par defaut pour permettre le controle camera.
 */
class Window {
public:
    GLFWwindow* handle;  ///< Pointeur vers la fenetre GLFW
    int width, height;   ///< Dimensions de la fenetre en pixels

    /**
     * @brief Constructeur par defaut
     * @details Initialise le handle a nullptr et les dimensions a zero.
     */
    Window() : handle(nullptr), width(0), height(0) {}

    /**
     * @brief Initialise la fenetre GLFW et le contexte OpenGL
     * @details Cette methode effectue plusieurs etapes :
     *          1. Initialisation de GLFW
     *          2. Creation de la fenetre avec un contexte OpenGL 2.1
     *          3. Initialisation de GLEW pour les extensions OpenGL
     *          4. Configuration du viewport et de la projection orthographique
     *          Le V-Sync est desactive (swap interval = 0) et le curseur est capture.
     * @param w Largeur de la fenetre en pixels
     * @param h Hauteur de la fenetre en pixels
     * @param title Titre affiche dans la barre de la fenetre
     * @return true si l'initialisation a reussi, false en cas d'erreur
     */
    bool initialize(int w, int h, const char* title) {
        width = w;
        height = h;

        if (!glfwInit()) {
            fprintf(stderr, "Failed to initialize GLFW\n");
            return false;
        }

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        handle = glfwCreateWindow(width, height, title, nullptr, nullptr);
        if (!handle) {
            fprintf(stderr, "Failed to create GLFW window\n");
            glfwTerminate();
            return false;
        }

        glfwMakeContextCurrent(handle);

        glewExperimental = GL_TRUE;
        GLenum glew_err = glewInit();
        if (glew_err != GLEW_OK) {
            fprintf(stderr, "Failed to initialize GLEW: %s\n", glewGetErrorString(glew_err));
            glfwDestroyWindow(handle);
            glfwTerminate();
            return false;
        }

        glfwSwapInterval(0);
        glfwSetInputMode(handle, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

        glViewport(0, 0, width, height);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(-1, 1, -1, 1, -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        return true;
    }

    /**
     * @brief Verifie si la fenetre doit etre fermee
     * @details Interroge GLFW pour savoir si l'utilisateur a demande la fermeture
     *          (par exemple en cliquant sur le bouton X de la fenetre).
     * @return true si la fenetre doit etre fermee, false sinon
     */
    bool should_close() const {
        return glfwWindowShouldClose(handle);
    }

    /**
     * @brief Echange les buffers avant/arriere (double buffering)
     * @details Affiche le contenu du back buffer a l'ecran. Doit etre appele
     *          a chaque fin de frame pour mettre a jour l'affichage.
     */
    void swap_buffers() {
        glfwSwapBuffers(handle);
    }

    /**
     * @brief Modifie le titre de la fenetre
     * @details Utile pour afficher des informations dynamiques comme le nombre
     *          de FPS ou le nombre d'echantillons accumules.
     * @param title Nouveau titre a afficher dans la barre de la fenetre
     */
    void set_title(const char* title) {
        glfwSetWindowTitle(handle, title);
    }

    /**
     * @brief Libere les ressources de la fenetre et termine GLFW
     * @details Detruit la fenetre GLFW et appelle glfwTerminate() pour liberer
     *          toutes les ressources associees a GLFW.
     */
    void cleanup() {
        if (handle) {
            glfwDestroyWindow(handle);
            handle = nullptr;
        }
        glfwTerminate();
    }

    /**
     * @brief Destructeur
     * @details Ne libere pas automatiquement les ressources. Il faut appeler
     *          cleanup() explicitement avant la destruction.
     */
    ~Window() {}
};

}

#endif
#endif
