#ifndef RAYTRACER_INTERACTIVE_INPUT_HANDLER_CUH
#define RAYTRACER_INTERACTIVE_INPUT_HANDLER_CUH

#ifdef ENABLE_INTERACTIVE

/**
 * @file input_handler.cuh
 * @brief Gestion des entrees clavier et souris via les callbacks GLFW
 * @details Ce fichier contient la structure InputState qui stocke l'etat de toutes les
 *          entrees utilisateur, ainsi que la classe InputHandler qui configure les callbacks
 *          GLFW pour capturer les evenements clavier et souris en temps reel.
 */

#include <GLFW/glfw3.h>

namespace rt {

/**
 * @struct InputState
 * @brief Stocke l'etat complet des entrees clavier et souris
 * @details Cette structure contient les booleens de deplacement (WASD, espace, ctrl),
 *          les positions et deltas de la souris, ainsi que des flags pour les actions
 *          speciales (capture d'ecran, fermeture, reset de l'accumulation).
 *          Elle est mise a jour par InputHandler et lue par CameraController.
 */
struct InputState {
    bool forward = false;    ///< Touche W enfoncee (avancer)
    bool backward = false;   ///< Touche S enfoncee (reculer)
    bool left = false;       ///< Touche A enfoncee (aller a gauche)
    bool right = false;      ///< Touche D enfoncee (aller a droite)
    bool up = false;         ///< Touche Espace/Q enfoncee (monter)
    bool down = false;       ///< Touche Ctrl/E enfoncee (descendre)

    bool fast = false;       ///< Touche Shift enfoncee (deplacement rapide)

    double mouse_x = 0.0;        ///< Position X actuelle du curseur
    double mouse_y = 0.0;        ///< Position Y actuelle du curseur
    double last_mouse_x = 0.0;   ///< Position X du curseur a la frame precedente
    double last_mouse_y = 0.0;   ///< Position Y du curseur a la frame precedente
    double mouse_delta_x = 0.0;  ///< Deplacement horizontal de la souris cette frame
    double mouse_delta_y = 0.0;  ///< Deplacement vertical de la souris cette frame
    bool first_mouse = true;     ///< Premiere lecture de la souris (pour eviter un saut initial)
    bool mouse_captured = true;  ///< La souris est-elle capturee par la fenetre

    bool should_close = false;          ///< L'utilisateur veut fermer l'application (Echap)
    bool screenshot_requested = false;  ///< Capture d'ecran demandee (P ou F12)
    bool reset_accumulation = false;    ///< Reinitialisation du buffer d'accumulation demandee (R)

    /**
     * @brief Calcule le deplacement de la souris depuis la derniere frame
     * @details Lors du premier appel, initialise les positions precedentes pour
     *          eviter un grand delta initial. Ensuite, calcule la difference entre
     *          la position actuelle et la position precedente. Le delta Y est inverse
     *          car les coordonnees ecran vont de haut en bas.
     */
    void update_mouse_delta() {
        if (first_mouse) {
            last_mouse_x = mouse_x;
            last_mouse_y = mouse_y;
            first_mouse = false;
        }
        mouse_delta_x = mouse_x - last_mouse_x;
        mouse_delta_y = last_mouse_y - mouse_y;
        last_mouse_x = mouse_x;
        last_mouse_y = mouse_y;
    }

    /**
     * @brief Verifie si un mouvement est en cours (clavier ou souris)
     * @details Retourne true si une touche de deplacement est enfoncee ou si
     *          la souris a bouge pendant cette frame (et est capturee).
     * @return true s'il y a du mouvement, false sinon
     */
    bool has_movement() const {
        return forward || backward || left || right || up || down ||
               (mouse_captured && (mouse_delta_x != 0.0 || mouse_delta_y != 0.0));
    }

    /**
     * @brief Remet a zero les deltas et les flags d'evenements ponctuels
     * @details Doit etre appele a la fin de chaque frame pour reinitialiser
     *          les valeurs qui ne doivent persister qu'une seule frame
     *          (deltas souris, screenshot, reset accumulation).
     */
    void clear_deltas() {
        mouse_delta_x = 0.0;
        mouse_delta_y = 0.0;
        reset_accumulation = false;
        screenshot_requested = false;
    }
};

/**
 * @class InputHandler
 * @brief Gere les callbacks GLFW pour capturer les evenements clavier et souris
 * @details Cette classe enregistre des callbacks lambda aupres de GLFW pour intercepter
 *          les evenements clavier, souris (position et boutons). Elle utilise le mecanisme
 *          de user pointer de GLFW pour acceder a l'instance depuis les callbacks statiques.
 *          L'etat des entrees est stocke dans le membre public 'state'.
 */
class InputHandler {
public:
    InputState state;  ///< Etat actuel des entrees, mis a jour par les callbacks

    /**
     * @brief Configure les callbacks GLFW pour la fenetre donnee
     * @details Enregistre trois callbacks :
     *          - Clavier : gere les touches WASD, Shift, Echap, P/F12, R
     *          - Position souris : met a jour les coordonnees du curseur
     *          - Boutons souris : clic droit pour basculer la capture du curseur
     *          Utilise glfwSetWindowUserPointer pour passer le pointeur this aux callbacks.
     * @param window Pointeur vers la fenetre GLFW a laquelle attacher les callbacks
     */
    void setup_callbacks(GLFWwindow* window) {
        glfwSetWindowUserPointer(window, this);

        glfwSetKeyCallback(window, [](GLFWwindow* win, int key, int scancode, int action, int mods) {
            auto* handler = static_cast<InputHandler*>(glfwGetWindowUserPointer(win));
            handler->key_callback(win, key, action);
        });

        glfwSetCursorPosCallback(window, [](GLFWwindow* win, double x, double y) {
            auto* handler = static_cast<InputHandler*>(glfwGetWindowUserPointer(win));
            handler->state.mouse_x = x;
            handler->state.mouse_y = y;
        });

        glfwSetMouseButtonCallback(window, [](GLFWwindow* win, int button, int action, int mods) {
            auto* handler = static_cast<InputHandler*>(glfwGetWindowUserPointer(win));
            if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
                handler->toggle_mouse_capture(win);
            }
        });
    }

    /**
     * @brief Interroge GLFW pour les nouveaux evenements et met a jour les deltas souris
     * @details Appelle glfwPollEvents() pour traiter tous les evenements en attente,
     *          puis recalcule le deplacement de la souris depuis la derniere frame.
     *          Doit etre appele une fois par frame au debut de la boucle de rendu.
     */
    void poll_events() {
        glfwPollEvents();
        state.update_mouse_delta();
    }

private:
    /**
     * @brief Callback de gestion des touches clavier
     * @details Associe chaque touche a un booleen dans InputState. Les touches de
     *          deplacement (WASD, Espace, Ctrl) sont actives tant qu'elles sont enfoncees.
     *          Les touches d'action (Echap, P, F12, R) ne reagissent qu'a l'appui initial.
     * @param window Pointeur vers la fenetre GLFW (non utilise directement)
     * @param key Code de la touche GLFW
     * @param action Type d'evenement (GLFW_PRESS, GLFW_RELEASE, GLFW_REPEAT)
     */
    void key_callback(GLFWwindow* window, int key, int action) {
        bool pressed = (action == GLFW_PRESS || action == GLFW_REPEAT);

        switch (key) {
            case GLFW_KEY_W:           state.forward = pressed; break;
            case GLFW_KEY_S:           state.backward = pressed; break;
            case GLFW_KEY_A:           state.left = pressed; break;
            case GLFW_KEY_D:           state.right = pressed; break;
            case GLFW_KEY_Q:
            case GLFW_KEY_SPACE:       state.up = pressed; break;
            case GLFW_KEY_E:
            case GLFW_KEY_LEFT_CONTROL: state.down = pressed; break;
            case GLFW_KEY_LEFT_SHIFT:  state.fast = pressed; break;
            case GLFW_KEY_ESCAPE:
                if (action == GLFW_PRESS) state.should_close = true;
                break;
            case GLFW_KEY_P:
            case GLFW_KEY_F12:
                if (action == GLFW_PRESS) state.screenshot_requested = true;
                break;
            case GLFW_KEY_R:
                if (action == GLFW_PRESS) state.reset_accumulation = true;
                break;
        }
    }

    /**
     * @brief Bascule la capture du curseur par la fenetre
     * @details Alterne entre le mode curseur capture (GLFW_CURSOR_DISABLED, pour le
     *          controle camera) et le mode curseur normal (GLFW_CURSOR_NORMAL, pour
     *          interagir avec l'interface). Reinitialise le flag first_mouse pour eviter
     *          un saut de camera lors de la reprise de la capture.
     * @param window Pointeur vers la fenetre GLFW
     */
    void toggle_mouse_capture(GLFWwindow* window) {
        state.mouse_captured = !state.mouse_captured;
        glfwSetInputMode(window, GLFW_CURSOR,
            state.mouse_captured ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
        state.first_mouse = true;
    }
};

}

#endif
#endif
