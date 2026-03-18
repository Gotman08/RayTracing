#ifndef RAYTRACER_GEOMETRY_SPHERE_CUH
#define RAYTRACER_GEOMETRY_SPHERE_CUH

/**
 * @file sphere.cuh
 * @brief Definition de la classe Sphere pour le lancer de rayons.
 * @details Ce fichier contient la classe Sphere qui represente une sphere 3D
 *          definie par son centre, son rayon et son materiau. L'intersection
 *          rayon-sphere est calculee en resolvant l'equation quadratique
 *          classique du ray tracing.
 */

#include "raytracer/geometry/hittable.cuh"
#include "raytracer/geometry/geometry_utils.cuh"
#include "raytracer/core/cuda_utils.cuh"

namespace rt {

/**
 * @class Sphere
 * @brief Represente une sphere dans la scene 3D.
 * @details La sphere est definie par son centre, son rayon et un materiau.
 *          Elle herite de Hittable et implemente la methode hit() qui resout
 *          l'equation quadratique d'intersection rayon-sphere. La boite englobante
 *          est calculee automatiquement dans le constructeur.
 */
class Sphere : public Hittable {
public:
    Point3 center;  ///< Centre de la sphere dans l'espace 3D
    float radius;   ///< Rayon de la sphere (toujours >= 0)
    Material* mat;  ///< Pointeur vers le materiau de la sphere

    /**
     * @brief Constructeur par defaut.
     * @details Cree une sphere de rayon 0 sans materiau.
     */
    __host__ __device__ Sphere() : Hittable(HittableType::SPHERE), radius(0), mat(nullptr) {}

    /**
     * @brief Constructeur parametrique de la sphere.
     * @details Initialise la sphere avec les parametres donnes et calcule
     *          automatiquement sa boite englobante (AABB). Le rayon est
     *          clamp a 0 minimum pour eviter les valeurs negatives.
     * @param c Centre de la sphere.
     * @param r Rayon de la sphere (sera mis a 0 si negatif).
     * @param m Pointeur vers le materiau a associer.
     */
    __host__ __device__ Sphere(const Point3& c, float r, Material* m)
        : Hittable(HittableType::SPHERE), center(c), radius(fmaxf(0.0f, r)), mat(m) {
        Vec3 rvec(radius, radius, radius);
        bbox = AABB(center - rvec, center + rvec);
    }

    /**
     * @brief Teste l'intersection entre un rayon et cette sphere.
     * @details On resout l'equation quadratique issue de la substitution de
     *          l'equation parametrique du rayon P(t) = O + t*D dans l'equation
     *          de la sphere |P - C|^2 = r^2. On utilise la formule optimisee
     *          avec h = dot(D, OC) pour eviter le facteur 2. Si le discriminant
     *          est negatif, il n'y a pas d'intersection. Sinon, on prend la
     *          racine la plus proche qui se trouve dans l'intervalle valide.
     *          Les coordonnees UV sont calculees pour le placage de texture.
     * @param r Le rayon a tester.
     * @param ray_t Intervalle de validite du parametre t.
     * @param rec Structure de sortie remplie avec les informations d'intersection.
     * @return true si une intersection valide est trouvee, false sinon.
     */
    __host__ __device__ bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const {
        Vec3 oc = center - r.origin();
        float a = r.direction().length_squared();
        float h = dot(r.direction(), oc);
        float c = oc.length_squared() - radius * radius;
        float discriminant = h * h - a * c;

        if (discriminant < 0)
            return false;

        float sqrtd = sqrtf(discriminant);

        float root = (h - sqrtd) / a;
        if (!ray_t.surrounds(root)) {
            root = (h + sqrtd) / a;
            if (!ray_t.surrounds(root))
                return false;
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        Vec3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);
        get_sphere_uv(outward_normal, rec.u, rec.v);
        rec.mat = mat;

        return true;
    }
};

}

#endif
