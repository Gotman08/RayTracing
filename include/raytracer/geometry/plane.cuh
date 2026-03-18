#ifndef RAYTRACER_GEOMETRY_PLANE_CUH
#define RAYTRACER_GEOMETRY_PLANE_CUH

/**
 * @file plane.cuh
 * @brief Definition de la classe Plane pour le lancer de rayons.
 * @details Ce fichier contient la classe Plane qui represente un plan infini
 *          dans la scene 3D. Le plan est defini par un point et un vecteur normal.
 *          L'intersection rayon-plan est calculee analytiquement et les coordonnees
 *          UV sont determinees par projection sur un repere tangent au plan.
 */

#include "raytracer/geometry/hittable.cuh"
#include "raytracer/core/cuda_utils.cuh"

namespace rt {

/**
 * @class Plane
 * @brief Represente un plan infini dans la scene 3D.
 * @details Le plan est defini par un point appartenant au plan, un vecteur normal
 *          et un materiau. Puisqu'un plan est infini, sa boite englobante est
 *          approximee par une tres grande boite orientee selon les vecteurs tangents.
 *          Les coordonnees UV sont calculees pour permettre le placage de texture
 *          sur le plan avec un motif repetitif.
 */
class Plane : public Hittable {
public:
    Point3 point;   ///< Un point appartenant au plan
    Vec3 normal;    ///< Vecteur normal unitaire au plan
    Material* mat;  ///< Pointeur vers le materiau du plan

    /**
     * @brief Constructeur par defaut.
     * @details Cree un plan sans materiau avec une normale et un point a l'origine.
     */
    __host__ __device__ Plane() : Hittable(HittableType::PLANE), mat(nullptr) {}

    /**
     * @brief Constructeur parametrique du plan.
     * @details Initialise le plan et calcule sa boite englobante. Comme un plan
     *          est infini, la boite englobante est approximee par une tres grande
     *          boite (1e10 unites) orientee selon les vecteurs tangents au plan.
     *          Les tangentes sont calculees par produit vectoriel avec un axe
     *          non parallele a la normale.
     * @param p Un point appartenant au plan.
     * @param n Vecteur normal au plan (sera normalise automatiquement).
     * @param m Pointeur vers le materiau a associer.
     */
    __host__ __device__ Plane(const Point3& p, const Vec3& n, Material* m)
        : Hittable(HittableType::PLANE), point(p), normal(unit_vector(n)), mat(m) {
        float big = 1e10f;
        Vec3 tangent1, tangent2;

        if (fabsf(normal.x) > 0.9f) {
            tangent1 = cross(normal, Vec3(0, 1, 0)).normalized();
        } else {
            tangent1 = cross(normal, Vec3(1, 0, 0)).normalized();
        }
        tangent2 = cross(normal, tangent1);

        Point3 min_p = point - big * tangent1 - big * tangent2 - EPSILON * normal;
        Point3 max_p = point + big * tangent1 + big * tangent2 + EPSILON * normal;
        bbox = AABB(min_p, max_p);
    }

    /**
     * @brief Teste l'intersection entre un rayon et ce plan infini.
     * @details On calcule le denominateur dot(normal, direction). S'il est trop
     *          proche de zero, le rayon est quasi-parallele au plan et on ignore
     *          l'intersection. Sinon, on calcule le parametre t et on verifie
     *          qu'il est dans l'intervalle valide. Les coordonnees UV sont ensuite
     *          calculees en projetant le point d'intersection sur le repere tangent
     *          du plan, avec un facteur d'echelle de 0.1 et un repliement (fract)
     *          pour obtenir des valeurs entre 0 et 1 (utile pour les textures repetees).
     * @param r Le rayon a tester.
     * @param ray_t Intervalle de validite du parametre t.
     * @param rec Structure de sortie remplie avec les informations d'intersection.
     * @return true si une intersection valide est trouvee, false sinon.
     */
    __host__ __device__ bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const {
        float denom = dot(normal, r.direction());

        if (fabsf(denom) < EPSILON)
            return false;

        float t = dot(point - r.origin(), normal) / denom;

        if (!ray_t.surrounds(t))
            return false;

        rec.t = t;
        rec.p = r.at(t);
        rec.set_face_normal(r, normal);
        rec.mat = mat;

        Vec3 local = rec.p - point;
        Vec3 tangent1, tangent2;
        if (fabsf(normal.x) > 0.9f) {
            tangent1 = cross(normal, Vec3(0, 1, 0)).normalized();
        } else {
            tangent1 = cross(normal, Vec3(1, 0, 0)).normalized();
        }
        tangent2 = cross(normal, tangent1);

        rec.u = dot(local, tangent1) * 0.1f;
        rec.v = dot(local, tangent2) * 0.1f;
        rec.u = rec.u - floorf(rec.u);
        rec.v = rec.v - floorf(rec.v);

        return true;
    }
};

}

#endif
