#ifndef RAYTRACER_INTERACTIVE_CAMERA_CONTROLLER_CUH
#define RAYTRACER_INTERACTIVE_CAMERA_CONTROLLER_CUH

#ifdef ENABLE_INTERACTIVE

/**
 * @file camera_controller.cuh
 * @brief Controleur de camera interactif avec deplacement WASD et orientation souris
 * @details Ce fichier contient la classe CameraController qui permet de deplacer la camera
 *          en temps reel dans la scene 3D. Le deplacement se fait avec les touches WASD
 *          (comme dans un FPS) et l'orientation est controlee par la souris via les angles
 *          de lacet (yaw) et de tangage (pitch).
 */

#include "raytracer/core/vec3.cuh"
#include "raytracer/camera/camera.cuh"
#include "raytracer/interactive/input_handler.cuh"
#include <cmath>

namespace rt {

/**
 * @class CameraController
 * @brief Gere les deplacements et l'orientation de la camera en mode interactif
 * @details Cette classe traduit les entrees clavier/souris en mouvements de camera.
 *          Elle utilise les angles d'Euler (yaw et pitch) pour l'orientation et permet
 *          un deplacement libre dans la scene. Le flag camera_changed indique si la
 *          camera a bouge depuis la derniere frame, ce qui est utile pour savoir
 *          s'il faut reinitialiser le buffer d'accumulation.
 */
class CameraController {
public:
    Point3 position;          ///< Position actuelle de la camera dans la scene
    float yaw;                ///< Angle de lacet (rotation horizontale) en radians
    float pitch;              ///< Angle de tangage (rotation verticale) en radians

    float vfov;               ///< Champ de vision vertical en degres
    float aperture;           ///< Ouverture de l'objectif pour le flou de profondeur
    float focus_dist;         ///< Distance de mise au point

    float move_speed;         ///< Vitesse de deplacement de la camera (unites par seconde)
    float mouse_sensitivity;  ///< Sensibilite de la souris pour la rotation

    bool camera_changed;      ///< Indique si la camera a ete modifiee cette frame

    /**
     * @brief Constructeur par defaut
     * @details Initialise la camera a l'origine avec des parametres par defaut raisonnables :
     *          FOV de 20 degres, pas de flou, vitesse de 5 unites/s.
     */
    CameraController()
        : position(0, 0, 0), yaw(0), pitch(0),
          vfov(20.0f), aperture(0.0f), focus_dist(10.0f),
          move_speed(5.0f), mouse_sensitivity(0.002f),
          camera_changed(true) {}

    /**
     * @brief Initialise le controleur avec une position et une cible
     * @details Calcule automatiquement les angles yaw et pitch a partir de la direction
     *          de regard (lookfrom vers lookat). Cela permet d'initialiser la camera
     *          de facon coherente avec la scene.
     * @param lookfrom Position initiale de la camera
     * @param lookat Point vers lequel la camera regarde initialement
     * @param fov Champ de vision vertical en degres
     * @param ap Ouverture de l'objectif (defaut : 0.0, pas de flou)
     * @param fd Distance de mise au point (defaut : 10.0)
     */
    void initialize(const Point3& lookfrom, const Point3& lookat,
                   float fov, float ap = 0.0f, float fd = 10.0f) {
        position = lookfrom;
        vfov = fov;
        aperture = ap;
        focus_dist = fd;

        Vec3 look_dir = unit_vector(lookat - lookfrom);
        yaw = atan2f(look_dir.x, -look_dir.z);
        pitch = asinf(fmaxf(-0.99f, fminf(0.99f, look_dir.y)));

        camera_changed = true;
    }

    /**
     * @brief Met a jour la position et l'orientation de la camera selon les entrees
     * @details Traite d'abord la rotation de la souris (yaw/pitch), puis le deplacement
     *          clavier (WASD + espace/ctrl). Le pitch est limite a environ +-86 degres
     *          pour eviter le gimbal lock. La touche Shift triple la vitesse de deplacement.
     * @param input Etat actuel des entrees clavier et souris
     * @param delta_time Temps ecoule depuis la derniere frame en secondes
     * @return true si la camera a change de position ou d'orientation, false sinon
     */
    bool update(const InputState& input, float delta_time) {
        camera_changed = false;

        if (input.mouse_captured) {
            if (input.mouse_delta_x != 0.0 || input.mouse_delta_y != 0.0) {
                yaw += static_cast<float>(input.mouse_delta_x) * mouse_sensitivity;
                pitch += static_cast<float>(input.mouse_delta_y) * mouse_sensitivity;

                constexpr float MAX_PITCH = 1.5f;
                pitch = fmaxf(-MAX_PITCH, fminf(MAX_PITCH, pitch));

                camera_changed = true;
            }
        }

        Vec3 forward = get_forward();
        Vec3 right = get_right();
        Vec3 up(0, 1, 0);

        Vec3 velocity(0, 0, 0);
        if (input.forward)  velocity = velocity + forward;
        if (input.backward) velocity = velocity - forward;
        if (input.right)    velocity = velocity + right;
        if (input.left)     velocity = velocity - right;
        if (input.up)       velocity = velocity + up;
        if (input.down)     velocity = velocity - up;

        if (velocity.length_squared() > 0.0f) {
            float speed = input.fast ? move_speed * 3.0f : move_speed;
            position = position + unit_vector(velocity) * speed * delta_time;
            camera_changed = true;
        }

        return camera_changed;
    }

    /**
     * @brief Calcule le vecteur direction avant de la camera
     * @details Utilise les angles yaw et pitch pour calculer la direction dans laquelle
     *          la camera regarde, en coordonnees cartesiennes (Y vers le haut).
     * @return Vecteur unitaire representant la direction avant
     */
    Vec3 get_forward() const {
        return Vec3(
            sinf(yaw) * cosf(pitch),
            sinf(pitch),
            -cosf(yaw) * cosf(pitch)
        );
    }

    /**
     * @brief Calcule le vecteur direction droite de la camera
     * @details Le vecteur droite est toujours horizontal (pas de composante Y),
     *          ce qui donne un deplacement lateral naturel.
     * @return Vecteur unitaire representant la direction droite
     */
    Vec3 get_right() const {
        return Vec3(cosf(yaw), 0, sinf(yaw));
    }

    /**
     * @brief Calcule le point vise par la camera (lookat)
     * @details Le point vise est situe a une distance focus_dist devant la camera
     *          dans la direction avant. Utilise pour construire l'objet Camera.
     * @return Point 3D vers lequel la camera regarde
     */
    Point3 get_lookat() const {
        return position + get_forward() * focus_dist;
    }

    /**
     * @brief Construit un objet Camera a partir de l'etat actuel du controleur
     * @details Cree et initialise un objet Camera utilisable pour le rendu,
     *          avec tous les parametres actuels (position, orientation, FOV, etc.).
     * @param width Largeur de l'image de rendu en pixels
     * @param height Hauteur de l'image de rendu en pixels
     * @return Objet Camera pret pour le rendu
     */
    Camera build_camera(int width, int height) const {
        Camera camera;
        camera.initialize(
            width, height,
            position,
            get_lookat(),
            Vec3(0, 1, 0),
            vfov, aperture, focus_dist
        );
        return camera;
    }
};

}

#endif
#endif
