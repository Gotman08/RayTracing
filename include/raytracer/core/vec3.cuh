#ifndef RAYTRACER_CORE_VEC3_CUH
#define RAYTRACER_CORE_VEC3_CUH

/**
 * @file vec3.cuh
 * @brief Definition de la classe Vec3 et des fonctions utilitaires associees
 * @details Ce fichier contient la classe Vec3 qui represente un vecteur 3D.
 *          Elle est utilisee pour modeliser des positions, des directions et
 *          des couleurs dans le raytracer. Toutes les methodes sont compatibles
 *          avec CUDA (host et device). Des fonctions de generation aleatoire
 *          sur GPU (curand) sont aussi fournies pour l'echantillonnage Monte Carlo.
 */

#include <cmath>
#include "raytracer/core/cuda_compat.cuh"

#ifdef __CUDACC__
#include <curand_kernel.h>
#endif

namespace rt {

/** @brief Constante Pi en simple precision */
constexpr float VEC3_PI = 3.14159265358979323846f;
/** @brief Constante 2*Pi en simple precision, utile pour les angles en radians */
constexpr float VEC3_TWO_PI = 2.0f * VEC3_PI;

/**
 * @class Vec3
 * @brief Vecteur 3D a composantes flottantes
 * @details Classe de base du raytracer servant a representer des positions
 *          (Point3), des directions ou des couleurs (Color) dans l'espace 3D.
 *          Tous les operateurs arithmetiques courants sont surcharges pour
 *          faciliter les calculs vectoriels. Compatible host/device CUDA.
 */
class Vec3 {
public:
    float x, y, z; /**< Composantes du vecteur (ou r, g, b pour les couleurs) */

    /**
     * @brief Constructeur par defaut, initialise le vecteur a (0, 0, 0)
     */
    __host__ __device__ Vec3() : x(0.0f), y(0.0f), z(0.0f) {}

    /**
     * @brief Constructeur a partir de trois composantes
     * @param e0 Composante x
     * @param e1 Composante y
     * @param e2 Composante z
     */
    __host__ __device__ Vec3(float e0, float e1, float e2) : x(e0), y(e1), z(e2) {}

    /**
     * @brief Constructeur explicite a partir d'une seule valeur
     * @details Initialise les trois composantes a la meme valeur v
     * @param v Valeur commune pour x, y et z
     */
    __host__ __device__ explicit Vec3(float v) : x(v), y(v), z(v) {}

    /**
     * @brief Operateur de negation unaire
     * @return Un nouveau vecteur dont chaque composante est l'opposee
     */
    __host__ __device__ Vec3 operator-() const { return Vec3(-x, -y, -z); }

    /**
     * @brief Acces a une composante par indice
     * @param i Indice de la composante (0 = x, 1 = y, 2 = z)
     * @return La valeur de la composante demandee
     */
    __host__ __device__ float operator[](int i) const {
        if (i == 0) return x;
        if (i == 1) return y;
        return z;
    }

    /**
     * @brief Addition de deux vecteurs composante par composante
     * @param v Le vecteur a additionner
     * @return Le vecteur somme
     */
    __host__ __device__ Vec3 operator+(const Vec3& v) const {
        return Vec3(x + v.x, y + v.y, z + v.z);
    }

    /**
     * @brief Soustraction de deux vecteurs composante par composante
     * @param v Le vecteur a soustraire
     * @return Le vecteur difference
     */
    __host__ __device__ Vec3 operator-(const Vec3& v) const {
        return Vec3(x - v.x, y - v.y, z - v.z);
    }

    /**
     * @brief Multiplication composante par composante (produit de Hadamard)
     * @details Utile pour la modulation des couleurs (ex: couleur * albedo)
     * @param v Le vecteur multiplicateur
     * @return Le vecteur resultat
     */
    __host__ __device__ Vec3 operator*(const Vec3& v) const {
        return Vec3(x * v.x, y * v.y, z * v.z);
    }

    /**
     * @brief Multiplication par un scalaire
     * @param t Le scalaire multiplicateur
     * @return Le vecteur mis a l'echelle
     */
    __host__ __device__ Vec3 operator*(float t) const {
        return Vec3(x * t, y * t, z * t);
    }

    /**
     * @brief Division par un scalaire
     * @param t Le scalaire diviseur
     * @return Le vecteur divise
     */
    __host__ __device__ Vec3 operator/(float t) const {
        return Vec3(x / t, y / t, z / t);
    }

    /**
     * @brief Addition en place d'un vecteur
     * @param v Le vecteur a ajouter
     * @return Reference vers le vecteur modifie
     */
    __host__ __device__ Vec3& operator+=(const Vec3& v) {
        x += v.x; y += v.y; z += v.z;
        return *this;
    }

    /**
     * @brief Multiplication en place par un scalaire
     * @param t Le scalaire multiplicateur
     * @return Reference vers le vecteur modifie
     */
    __host__ __device__ Vec3& operator*=(float t) {
        x *= t; y *= t; z *= t;
        return *this;
    }

    /**
     * @brief Division en place par un scalaire
     * @param t Le scalaire diviseur
     * @return Reference vers le vecteur modifie
     */
    __host__ __device__ Vec3& operator/=(float t) {
        return *this *= 1.0f / t;
    }

    /**
     * @brief Calcule la norme (longueur) du vecteur
     * @return La norme euclidienne du vecteur
     */
    __host__ __device__ float length() const {
        return sqrtf(x*x + y*y + z*z);
    }

    /**
     * @brief Calcule le carre de la norme du vecteur
     * @details Plus rapide que length() car evite la racine carree.
     *          Utile pour les comparaisons de distances.
     * @return Le carre de la norme euclidienne
     */
    __host__ __device__ float length_squared() const {
        return x*x + y*y + z*z;
    }

    /**
     * @brief Retourne le vecteur normalise (de norme 1)
     * @details Si le vecteur est de norme nulle, retourne le vecteur nul
     *          pour eviter une division par zero.
     * @return Le vecteur unitaire dans la meme direction
     */
    __host__ __device__ Vec3 normalized() const {
        float len = length();
        return (len > 0.0f) ? (*this / len) : Vec3();
    }

    /**
     * @brief Verifie si le vecteur est proche du vecteur nul
     * @details Chaque composante est comparee a un seuil de 1e-8.
     *          Utile pour eviter des erreurs numeriques lors des reflexions.
     * @return true si toutes les composantes sont proches de zero
     */
    __host__ __device__ bool near_zero() const {
        const float s = 1e-8f;
        return (fabsf(x) < s) && (fabsf(y) < s) && (fabsf(z) < s);
    }
};

/** @brief Alias de type : un Point3 est un Vec3 utilise comme position */
using Point3 = Vec3;
/** @brief Alias de type : une Color est un Vec3 utilise comme couleur RGB */
using Color = Vec3;

/**
 * @brief Multiplication scalaire * vecteur (commutativite)
 * @param t Le scalaire
 * @param v Le vecteur
 * @return Le vecteur mis a l'echelle
 */
__host__ __device__ inline Vec3 operator*(float t, const Vec3& v) {
    return v * t;
}

/**
 * @brief Calcule le produit scalaire de deux vecteurs
 * @details Le produit scalaire donne le cosinus de l'angle entre les vecteurs
 *          (si ceux-ci sont normalises). Tres utilise en eclairage.
 * @param u Premier vecteur
 * @param v Deuxieme vecteur
 * @return Le produit scalaire u . v
 */
__host__ __device__ inline float dot(const Vec3& u, const Vec3& v) {
    return u.x*v.x + u.y*v.y + u.z*v.z;
}

/**
 * @brief Calcule le produit vectoriel de deux vecteurs
 * @details Le resultat est un vecteur perpendiculaire au plan forme par u et v.
 *          Utilise pour calculer les normales de surfaces.
 * @param u Premier vecteur
 * @param v Deuxieme vecteur
 * @return Le vecteur u x v
 */
__host__ __device__ inline Vec3 cross(const Vec3& u, const Vec3& v) {
    return Vec3(
        u.y*v.z - u.z*v.y,
        u.z*v.x - u.x*v.z,
        u.x*v.y - u.y*v.x
    );
}

/**
 * @brief Calcule la reflexion d'un vecteur par rapport a une normale
 * @details Applique la formule de reflexion : r = v - 2*(v.n)*n.
 *          Utilisee pour les materiaux metalliques et les miroirs.
 * @param v Le vecteur incident
 * @param n La normale a la surface (doit etre unitaire)
 * @return Le vecteur reflechi
 */
__host__ __device__ inline Vec3 reflect(const Vec3& v, const Vec3& n) {
    return v - 2.0f * dot(v, n) * n;
}

/**
 * @brief Calcule la refraction d'un vecteur selon la loi de Snell-Descartes
 * @details Decompose le rayon refracte en composantes perpendiculaire et
 *          parallele a la normale. Utilisee pour les materiaux dielectriques
 *          (verre, eau, etc.).
 * @param uv Le vecteur incident (doit etre unitaire)
 * @param n La normale a la surface (doit etre unitaire)
 * @param etai_over_etat Le rapport des indices de refraction (n1/n2)
 * @return Le vecteur refracte
 */
__host__ __device__ inline Vec3 refract(const Vec3& uv, const Vec3& n, float etai_over_etat) {
    float cos_theta = fminf(dot(-uv, n), 1.0f);
    Vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    Vec3 r_out_parallel = -sqrtf(fabsf(1.0f - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

/**
 * @brief Retourne le vecteur unitaire (normalise) d'un vecteur donne
 * @param v Le vecteur a normaliser
 * @return Le vecteur unitaire dans la meme direction que v
 */
__host__ __device__ inline Vec3 unit_vector(const Vec3& v) {
    return v.normalized();
}

#ifdef __CUDACC__
/**
 * @brief Genere un vecteur 3D aleatoire avec des composantes dans [0, 1]
 * @details Utilise le generateur curand sur GPU. Chaque composante est
 *          independante et uniformement distribuee.
 * @param rand_state Etat du generateur aleatoire CUDA
 * @return Un vecteur avec x, y, z dans [0, 1]
 */
__device__ inline Vec3 random_vec3(curandState* rand_state) {
    return Vec3(
        curand_uniform(rand_state),
        curand_uniform(rand_state),
        curand_uniform(rand_state)
    );
}

/**
 * @brief Genere un vecteur 3D aleatoire avec des composantes dans [min, max]
 * @param min Borne inferieure de l'intervalle
 * @param max Borne superieure de l'intervalle
 * @param rand_state Etat du generateur aleatoire CUDA
 * @return Un vecteur avec x, y, z dans [min, max]
 */
__device__ inline Vec3 random_vec3(float min, float max, curandState* rand_state) {
    float range = max - min;
    return Vec3(
        min + range * curand_uniform(rand_state),
        min + range * curand_uniform(rand_state),
        min + range * curand_uniform(rand_state)
    );
}

/**
 * @brief Genere un vecteur unitaire aleatoire uniformement distribue sur la sphere
 * @details Utilise la methode des coordonnees spheriques pour obtenir une
 *          distribution uniforme sur la surface de la sphere unite.
 *          Essentiel pour l'echantillonnage diffus (lambertien).
 * @param rand_state Etat du generateur aleatoire CUDA
 * @return Un vecteur de norme 1, direction aleatoire uniforme
 */
__device__ inline Vec3 random_unit_vector(curandState* rand_state) {
    float z = 2.0f * curand_uniform(rand_state) - 1.0f;
    float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    float phi = VEC3_TWO_PI * curand_uniform(rand_state);
    return Vec3(r * cosf(phi), r * sinf(phi), z);
}

/**
 * @brief Genere un point aleatoire a l'interieur de la sphere unite
 * @details Combine un vecteur unitaire aleatoire avec un rayon distribue
 *          selon la racine cubique pour obtenir une distribution volumique
 *          uniforme dans la sphere.
 * @param rand_state Etat du generateur aleatoire CUDA
 * @return Un vecteur de norme <= 1
 */
__device__ inline Vec3 random_in_unit_sphere(curandState* rand_state) {
    Vec3 dir = random_unit_vector(rand_state);
    float r = cbrtf(curand_uniform(rand_state));
    return dir * r;
}

/**
 * @brief Genere un vecteur aleatoire dans l'hemisphere defini par une normale
 * @details Si le vecteur aleatoire est du mauvais cote de la surface
 *          (produit scalaire negatif avec la normale), on l'inverse.
 *          Utilise pour l'echantillonnage hemispherique.
 * @param normal La normale de la surface definissant l'hemisphere
 * @param rand_state Etat du generateur aleatoire CUDA
 * @return Un vecteur unitaire dans l'hemisphere de la normale
 */
__device__ inline Vec3 random_on_hemisphere(const Vec3& normal, curandState* rand_state) {
    Vec3 on_unit_sphere = random_unit_vector(rand_state);
    if (dot(on_unit_sphere, normal) > 0.0f)
        return on_unit_sphere;
    else
        return -on_unit_sphere;
}

/**
 * @brief Genere un point aleatoire dans le disque unite (plan z = 0)
 * @details Utilise pour simuler l'ouverture de l'objectif de la camera
 *          (effet de profondeur de champ / defocus blur).
 * @param rand_state Etat du generateur aleatoire CUDA
 * @return Un vecteur (x, y, 0) de norme <= 1
 */
__device__ inline Vec3 random_in_unit_disk(curandState* rand_state) {
    float r = sqrtf(curand_uniform(rand_state));
    float theta = VEC3_TWO_PI * curand_uniform(rand_state);
    return Vec3(r * cosf(theta), r * sinf(theta), 0.0f);
}

/**
 * @brief Genere une direction aleatoire selon une distribution cosinus
 * @details La probabilite est proportionnelle au cosinus de l'angle avec
 *          la normale (axe z local). C'est la distribution optimale pour
 *          l'eclairage lambertien car elle reduit la variance de l'estimateur
 *          Monte Carlo.
 * @param rand_state Etat du generateur aleatoire CUDA
 * @return Un vecteur dans l'hemisphere z > 0, distribue selon le cosinus
 */
__device__ inline Vec3 random_cosine_direction(curandState* rand_state) {
    float r1 = curand_uniform(rand_state);
    float r2 = curand_uniform(rand_state);

    float phi = 2.0f * 3.14159265358979323846f * r1;
    float x = cosf(phi) * sqrtf(r2);
    float y = sinf(phi) * sqrtf(r2);
    float z = sqrtf(1.0f - r2);

    return Vec3(x, y, z);
}
#endif

}

#endif
