/**
 * @file test_materials.cpp
 * @brief Tests unitaires pour les materiaux du raytracer.
 *
 * Ce fichier couvre les tests de creation des materiaux (lambertien, metal, dielectrique),
 * le comportement de diffusion (scatter) de chaque type de materiau,
 * la verification de l'attenuation, et l'approximation de Schlick pour la reflectance.
 */

#include <iostream>
#include <cmath>

#include "raytracer/core/vec3.cuh"
#include "raytracer/core/ray.cuh"
#include "raytracer/core/hit_record.cuh"
#include "raytracer/core/random.cuh"

#ifndef __CUDACC__
inline float curand_uniform(curandState*) { return 0.5f; }
namespace rt {
    inline Vec3 random_unit_vector(curandState*) { return Vec3(0, 1, 0); }
    inline Vec3 random_in_unit_sphere(curandState*) { return Vec3(0, 0, 0); }
    inline Vec3 random_in_unit_disk(curandState*) { return Vec3(0, 0, 0); }
}
#endif

#include "raytracer/materials/material.cuh"
#include "raytracer/materials/lambertian.cuh"
#include "raytracer/materials/metal.cuh"
#include "raytracer/materials/dielectric.cuh"

using namespace rt;

/**
 * @brief Macro d'assertion simple pour les tests.
 *
 * Verifie qu'une expression est vraie, sinon affiche un message d'erreur
 * et retourne false pour signaler l'echec.
 */
#define TEST_ASSERT(expr) \
    do { \
        if (!(expr)) { \
            std::cerr << "    FAILED: " << #expr << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            return false; \
        } \
    } while(0)

/**
 * @brief Macro d'assertion avec tolerance pour comparer des valeurs flottantes.
 */
#define TEST_ASSERT_NEAR(a, b, eps) \
    do { \
        if (std::abs((a) - (b)) > (eps)) { \
            std::cerr << "    FAILED: " << #a << " (" << (a) << ") != " << #b << " (" << (b) << ")" \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            return false; \
        } \
    } while(0)

/**
 * @brief Macro d'execution d'un test avec affichage du resultat et mise a jour des compteurs.
 */
#define RUN_TEST(test_func) \
    do { \
        if (test_func()) { \
            std::cout << "  [PASS] " << #test_func << "\n"; \
            passed++; \
        } else { \
            std::cout << "  [FAIL] " << #test_func << "\n"; \
            failed++; \
        } \
        total++; \
    } while(0)


/**
 * @brief Verifie la creation d'un materiau lambertien avec un albedo donne.
 *
 * On s'assure que le type est bien LAMBERTIAN et que la couleur d'albedo
 * correspond aux valeurs fournies.
 */
static bool test_create_lambertian() {
    Material mat = create_lambertian(Color(0.5f, 0.3f, 0.1f));
    TEST_ASSERT(mat.type == MaterialType::LAMBERTIAN);
    TEST_ASSERT_NEAR(mat.albedo.x, 0.5f, 1e-5f);
    TEST_ASSERT_NEAR(mat.albedo.y, 0.3f, 1e-5f);
    TEST_ASSERT_NEAR(mat.albedo.z, 0.1f, 1e-5f);
    return true;
}

/**
 * @brief Verifie la creation d'un materiau metallique avec albedo et facteur de flou (fuzz).
 */
static bool test_create_metal() {
    Material mat = create_metal(Color(0.8f, 0.8f, 0.8f), 0.3f);
    TEST_ASSERT(mat.type == MaterialType::METAL);
    TEST_ASSERT_NEAR(mat.albedo.x, 0.8f, 1e-5f);
    TEST_ASSERT_NEAR(mat.fuzz, 0.3f, 1e-5f);
    return true;
}

/**
 * @brief Verifie que le facteur de flou (fuzz) d'un metal est clamp a 1.0 maximum.
 *
 * Meme si on passe une valeur de fuzz de 5.0, elle doit etre ramenee a 1.0.
 */
static bool test_create_metal_fuzz_clamp() {
    Material mat = create_metal(Color(0.8f, 0.8f, 0.8f), 5.0f);
    TEST_ASSERT(mat.fuzz <= 1.0f);
    return true;
}

/**
 * @brief Verifie la creation d'un materiau dielectrique (verre) avec un indice de refraction.
 *
 * L'albedo d'un dielectrique doit etre blanc (1, 1, 1) car le verre
 * ne colore pas la lumiere.
 */
static bool test_create_dielectric() {
    Material mat = create_dielectric(1.5f);
    TEST_ASSERT(mat.type == MaterialType::DIELECTRIC);
    TEST_ASSERT_NEAR(mat.ior, 1.5f, 1e-5f);
    TEST_ASSERT_NEAR(mat.albedo.x, 1.0f, 1e-5f);
    TEST_ASSERT_NEAR(mat.albedo.y, 1.0f, 1e-5f);
    TEST_ASSERT_NEAR(mat.albedo.z, 1.0f, 1e-5f);
    return true;
}

/**
 * @brief Verifie les valeurs par defaut d'un materiau non initialise.
 *
 * Par defaut, un materiau doit etre lambertien, avec un fuzz a 0 et un IOR a 1.5.
 */
static bool test_material_default() {
    Material mat;
    TEST_ASSERT(mat.type == MaterialType::LAMBERTIAN);
    TEST_ASSERT_NEAR(mat.fuzz, 0.0f, 1e-5f);
    TEST_ASSERT_NEAR(mat.ior, 1.5f, 1e-5f);
    return true;
}

/**
 * @brief Verifie la methode statique make_dielectric de la classe Material.
 *
 * On teste avec un indice de refraction de 2.4 (diamant) pour verifier
 * que le type et l'IOR sont correctement assignes.
 */
static bool test_make_dielectric_static() {
    Material mat = Material::make_dielectric(2.4f);
    TEST_ASSERT(mat.type == MaterialType::DIELECTRIC);
    TEST_ASSERT_NEAR(mat.ior, 2.4f, 1e-5f);
    return true;
}


/**
 * @brief Verifie que la diffusion lambertienne retourne toujours true.
 *
 * Un materiau lambertien diffuse toujours la lumiere (il ne l'absorbe jamais
 * completement), donc scatter doit toujours renvoyer true.
 */
static bool test_lambertian_scatter_always_true() {
    Material mat = create_lambertian(Color(0.5f, 0.5f, 0.5f));
    Ray r_in(Point3(0, 0, -1), Vec3(0, 0, 1));
    HitRecord rec;
    rec.p = Point3(0, 0, 0);
    rec.normal = Vec3(0, 0, -1);
    rec.front_face = true;
    rec.mat = &mat;

    Color attenuation;
    Ray scattered(Point3(0,0,0), Vec3(0,0,0));
    CPURandom rng(42);

    bool result = scatter_lambertian_cpu(mat, r_in, rec, attenuation, scattered, rng);
    TEST_ASSERT(result == true);
    return true;
}

/**
 * @brief Verifie que l'attenuation d'un materiau lambertien correspond a son albedo.
 *
 * Apres diffusion, la couleur d'attenuation doit etre egale a l'albedo
 * du materiau (c'est la fraction de lumiere reflechie par composante).
 */
static bool test_lambertian_attenuation_is_albedo() {
    Color albedo(0.7f, 0.3f, 0.1f);
    Material mat = create_lambertian(albedo);
    Ray r_in(Point3(0, 0, -1), Vec3(0, 0, 1));
    HitRecord rec;
    rec.p = Point3(0, 0, 0);
    rec.normal = Vec3(0, 0, -1);
    rec.front_face = true;

    Color attenuation;
    Ray scattered(Point3(0,0,0), Vec3(0,0,0));
    CPURandom rng(42);

    scatter_lambertian_cpu(mat, r_in, rec, attenuation, scattered, rng);
    TEST_ASSERT_NEAR(attenuation.x, 0.7f, 1e-5f);
    TEST_ASSERT_NEAR(attenuation.y, 0.3f, 1e-5f);
    TEST_ASSERT_NEAR(attenuation.z, 0.1f, 1e-5f);
    return true;
}

/**
 * @brief Verifie que le rayon diffuse part bien du point d'impact.
 *
 * L'origine du rayon diffuse doit etre exactement au point d'intersection
 * (hit point) sur la surface de l'objet.
 */
static bool test_lambertian_scatter_origin_at_hit() {
    Material mat = create_lambertian(Color(0.5f, 0.5f, 0.5f));
    Ray r_in(Point3(0, 0, -1), Vec3(0, 0, 1));
    HitRecord rec;
    rec.p = Point3(1, 2, 3);
    rec.normal = Vec3(0, 1, 0);
    rec.front_face = true;

    Color attenuation;
    Ray scattered(Point3(0,0,0), Vec3(0,0,0));
    CPURandom rng(42);

    scatter_lambertian_cpu(mat, r_in, rec, attenuation, scattered, rng);
    TEST_ASSERT_NEAR(scattered.origin().x, 1.0f, 1e-5f);
    TEST_ASSERT_NEAR(scattered.origin().y, 2.0f, 1e-5f);
    TEST_ASSERT_NEAR(scattered.origin().z, 3.0f, 1e-5f);
    return true;
}


/**
 * @brief Verifie que la reflexion metallique renvoie le rayon dans la bonne direction.
 *
 * Un rayon arrivant verticalement vers le bas sur une surface horizontale
 * doit etre reflechi vers le haut (composante Y positive).
 * Le fuzz est a 0 pour une reflexion parfaite.
 */
static bool test_metal_scatter_reflection_direction() {
    Material mat = create_metal(Color(0.9f, 0.9f, 0.9f), 0.0f);
    Ray r_in(Point3(0, 1, -1), Vec3(0, -1, 0));
    HitRecord rec;
    rec.p = Point3(0, 0, 0);
    rec.normal = Vec3(0, 1, 0);
    rec.front_face = true;

    Color attenuation;
    Ray scattered(Point3(0,0,0), Vec3(0,0,0));
    CPURandom rng(42);

    bool result = scatter_metal_cpu(mat, r_in, rec, attenuation, scattered, rng);
    TEST_ASSERT(result == true);
    TEST_ASSERT(scattered.direction().y > 0);
    return true;
}

/**
 * @brief Verifie que l'attenuation d'un materiau metallique correspond a son albedo.
 */
static bool test_metal_attenuation_is_albedo() {
    Color albedo(0.8f, 0.6f, 0.2f);
    Material mat = create_metal(albedo, 0.0f);
    Ray r_in(Point3(0, 1, 0), Vec3(0, -1, 0));
    HitRecord rec;
    rec.p = Point3(0, 0, 0);
    rec.normal = Vec3(0, 1, 0);
    rec.front_face = true;

    Color attenuation;
    Ray scattered(Point3(0,0,0), Vec3(0,0,0));
    CPURandom rng(42);

    scatter_metal_cpu(mat, r_in, rec, attenuation, scattered, rng);
    TEST_ASSERT_NEAR(attenuation.x, 0.8f, 1e-5f);
    TEST_ASSERT_NEAR(attenuation.y, 0.6f, 1e-5f);
    TEST_ASSERT_NEAR(attenuation.z, 0.2f, 1e-5f);
    return true;
}


/**
 * @brief Verifie l'approximation de Schlick a incidence normale (cosinus = 1).
 *
 * A incidence normale, la reflectance doit correspondre a R0 = ((1 - n) / (1 + n))^2,
 * soit environ 0.04 pour un IOR de 1.5 (verre classique).
 */
static bool test_schlick_reflectance_at_zero() {
    float r = reflectance(1.0f, 1.5f);
    float r0 = (1.0f - 1.5f) / (1.0f + 1.5f);
    r0 = r0 * r0;
    TEST_ASSERT_NEAR(r, r0, 1e-5f);
    return true;
}

/**
 * @brief Verifie que la reflectance de Schlick tend vers 1 en incidence rasante.
 *
 * Quand le cosinus est proche de 0 (angle rasant), presque toute la lumiere
 * est reflechie : la reflectance doit depasser 0.9.
 */
static bool test_schlick_reflectance_at_grazing() {
    float r = reflectance(0.01f, 1.5f);
    TEST_ASSERT(r > 0.9f);
    return true;
}

/**
 * @brief Verifie que la diffusion dielectrique retourne toujours true.
 *
 * Un materiau dielectrique (verre) refracte ou reflechit toujours le rayon,
 * il ne l'absorbe jamais.
 */
static bool test_dielectric_scatter_always_true() {
    Material mat = create_dielectric(1.5f);
    Ray r_in(Point3(0, 0, -2), Vec3(0, 0, 1));
    HitRecord rec;
    rec.p = Point3(0, 0, 0);
    rec.normal = Vec3(0, 0, -1);
    rec.front_face = true;
    rec.mat = &mat;

    Color attenuation;
    Ray scattered(Point3(0,0,0), Vec3(0,0,0));
    CPURandom rng(42);

    bool result = scatter_dielectric_cpu(mat, r_in, rec, attenuation, scattered, rng);
    TEST_ASSERT(result == true);
    return true;
}

/**
 * @brief Verifie que l'attenuation d'un dielectrique est blanche (1, 1, 1).
 *
 * Le verre ideal ne colore pas la lumiere, donc l'attenuation doit etre (1, 1, 1).
 */
static bool test_dielectric_attenuation_is_white() {
    Material mat = create_dielectric(1.5f);
    Ray r_in(Point3(0, 0, -2), Vec3(0, 0, 1));
    HitRecord rec;
    rec.p = Point3(0, 0, 0);
    rec.normal = Vec3(0, 0, -1);
    rec.front_face = true;

    Color attenuation;
    Ray scattered(Point3(0,0,0), Vec3(0,0,0));
    CPURandom rng(42);

    scatter_dielectric_cpu(mat, r_in, rec, attenuation, scattered, rng);
    TEST_ASSERT_NEAR(attenuation.x, 1.0f, 1e-5f);
    TEST_ASSERT_NEAR(attenuation.y, 1.0f, 1e-5f);
    TEST_ASSERT_NEAR(attenuation.z, 1.0f, 1e-5f);
    return true;
}


/**
 * @brief Lance l'ensemble des tests unitaires des materiaux.
 *
 * @param passed Compteur de tests reussis (mis a jour par reference).
 * @param failed Compteur de tests echoues (mis a jour par reference).
 * @param total  Compteur total de tests executes (mis a jour par reference).
 */
void run_materials_tests(int& passed, int& failed, int& total) {
    RUN_TEST(test_create_lambertian);
    RUN_TEST(test_create_metal);
    RUN_TEST(test_create_metal_fuzz_clamp);
    RUN_TEST(test_create_dielectric);
    RUN_TEST(test_material_default);
    RUN_TEST(test_make_dielectric_static);
    RUN_TEST(test_lambertian_scatter_always_true);
    RUN_TEST(test_lambertian_attenuation_is_albedo);
    RUN_TEST(test_lambertian_scatter_origin_at_hit);
    RUN_TEST(test_metal_scatter_reflection_direction);
    RUN_TEST(test_metal_attenuation_is_albedo);
    RUN_TEST(test_schlick_reflectance_at_zero);
    RUN_TEST(test_schlick_reflectance_at_grazing);
    RUN_TEST(test_dielectric_scatter_always_true);
    RUN_TEST(test_dielectric_attenuation_is_white);
}
