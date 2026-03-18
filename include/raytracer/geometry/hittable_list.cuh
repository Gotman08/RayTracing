#ifndef RAYTRACER_GEOMETRY_HITTABLE_LIST_CUH
#define RAYTRACER_GEOMETRY_HITTABLE_LIST_CUH

/**
 * @file hittable_list.cuh
 * @brief Conteneur pour stocker et gerer l'ensemble des objets de la scene.
 * @details Ce fichier definit les structures necessaires pour stocker des objets
 *          heterogenes (spheres, plans) dans un tableau unique compatible GPU.
 *          On utilise une union (HittableData) pour eviter l'allocation dynamique
 *          et un struct (HittableObject) pour associer chaque objet a son type et
 *          sa boite englobante. La classe HittableList gere la collection complete.
 */

#include "raytracer/geometry/hittable.cuh"
#include "raytracer/geometry/sphere.cuh"
#include "raytracer/geometry/plane.cuh"

namespace rt {

/**
 * @union HittableData
 * @brief Union qui stocke les donnees concretes d'un objet geometrique.
 * @details Comme on ne peut pas utiliser le polymorphisme dynamique sur GPU,
 *          on utilise une union pour stocker soit une Sphere, soit un Plane
 *          dans le meme espace memoire. Un seul des deux membres est valide
 *          a la fois, selon le type indique dans le HittableObject parent.
 */
union HittableData {
    Sphere sphere; ///< Donnees de la sphere (si le type est SPHERE)
    Plane plane;   ///< Donnees du plan (si le type est PLANE)

    /**
     * @brief Constructeur par defaut.
     * @details Ne fait rien volontairement car l'initialisation depend du type choisi.
     */
    __host__ __device__ HittableData() {}
};

/**
 * @struct HittableObject
 * @brief Represente un objet complet de la scene avec son type, ses donnees et sa boite englobante.
 * @details Cette structure combine le type de l'objet, ses donnees geometriques (via l'union)
 *          et sa boite englobante AABB. La methode hit() redirige le test d'intersection
 *          vers la bonne forme geometrique en fonction du type.
 */
struct HittableObject {
    HittableType type; ///< Type de l'objet (SPHERE ou PLANE)
    HittableData data; ///< Donnees geometriques de l'objet
    AABB bbox;         ///< Boite englobante de l'objet

    /**
     * @brief Constructeur par defaut.
     * @details Initialise le type a SPHERE par defaut.
     */
    __host__ __device__ HittableObject() : type(HittableType::SPHERE) {}

    /**
     * @brief Teste l'intersection entre un rayon et cet objet.
     * @details Effectue un switch sur le type de l'objet pour appeler la methode
     *          hit() appropriee (sphere ou plan). C'est l'equivalent d'un dispatch
     *          virtuel, mais compatible avec CUDA.
     * @param r Le rayon a tester.
     * @param ray_t Intervalle de validite du parametre t le long du rayon.
     * @param rec Structure de sortie contenant les informations d'intersection.
     * @return true si le rayon intersecte l'objet dans l'intervalle donne, false sinon.
     */
    __host__ __device__ bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const {
        switch (type) {
            case HittableType::SPHERE:
                return data.sphere.hit(r, ray_t, rec);
            case HittableType::PLANE:
                return data.plane.hit(r, ray_t, rec);
            default:
                return false;
        }
    }
};

/**
 * @class HittableList
 * @brief Liste de tous les objets de la scene.
 * @details Cette classe gere un tableau dynamique d'objets HittableObject.
 *          Elle fournit des methodes pour ajouter des spheres et des plans,
 *          ainsi qu'une methode hit() qui parcourt tous les objets pour trouver
 *          l'intersection la plus proche. La boite englobante globale est mise
 *          a jour automatiquement a chaque ajout d'objet.
 */
class HittableList {
public:
    HittableObject* objects; ///< Tableau d'objets de la scene
    int count;               ///< Nombre actuel d'objets dans la liste
    int capacity;            ///< Capacite maximale du tableau
    AABB bbox;               ///< Boite englobante englobant tous les objets

    /**
     * @brief Constructeur par defaut.
     * @details Initialise une liste vide sans memoire allouee.
     */
    __host__ __device__ HittableList() : objects(nullptr), count(0), capacity(0) {}

    /**
     * @brief Ajoute une sphere a la liste des objets.
     * @details Cree une sphere avec les parametres donnes, calcule sa boite englobante
     *          et met a jour la boite englobante globale de la scene. Si la capacite
     *          maximale est atteinte, l'ajout est ignore silencieusement.
     * @param center Centre de la sphere.
     * @param radius Rayon de la sphere.
     * @param mat Pointeur vers le materiau associe a la sphere.
     */
    __host__ __device__ void add_sphere(const Point3& center, float radius, Material* mat) {
        if (count >= capacity) return;
        objects[count].type = HittableType::SPHERE;
        objects[count].data.sphere = Sphere(center, radius, mat);
        objects[count].bbox = objects[count].data.sphere.bounding_box();
        bbox = (count == 0) ? objects[count].bbox : AABB(bbox, objects[count].bbox);
        count++;
    }

    /**
     * @brief Ajoute un plan infini a la liste des objets.
     * @details Cree un plan avec les parametres donnes, calcule sa boite englobante
     *          (approximee par une tres grande boite) et met a jour la boite englobante
     *          globale de la scene. Si la capacite maximale est atteinte, l'ajout est ignore.
     * @param point Un point appartenant au plan.
     * @param normal Le vecteur normal au plan.
     * @param mat Pointeur vers le materiau associe au plan.
     */
    __host__ __device__ void add_plane(const Point3& point, const Vec3& normal, Material* mat) {
        if (count >= capacity) return;
        objects[count].type = HittableType::PLANE;
        objects[count].data.plane = Plane(point, normal, mat);
        objects[count].bbox = objects[count].data.plane.bounding_box();
        bbox = (count == 0) ? objects[count].bbox : AABB(bbox, objects[count].bbox);
        count++;
    }

    /**
     * @brief Teste l'intersection d'un rayon avec tous les objets de la scene.
     * @details Parcourt lineairement tous les objets et conserve l'intersection
     *          la plus proche de l'origine du rayon. A chaque intersection trouvee,
     *          l'intervalle de recherche est reduit pour ne garder que les objets
     *          encore plus proches.
     * @param r Le rayon a tester.
     * @param ray_t Intervalle de validite du parametre t le long du rayon.
     * @param rec Structure de sortie contenant les informations de l'intersection la plus proche.
     * @return true si le rayon intersecte au moins un objet, false sinon.
     */
    __host__ __device__ bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const {
        HitRecord temp_rec;
        bool hit_anything = false;
        float closest_so_far = ray_t.max;

        for (int i = 0; i < count; i++) {
            if (objects[i].hit(r, Interval(ray_t.min, closest_so_far), temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }

    /**
     * @brief Retourne la boite englobante globale de la scene.
     * @return Reference constante vers l'AABB englobant tous les objets.
     */
    __host__ __device__ const AABB& bounding_box() const { return bbox; }
};

}

#endif
