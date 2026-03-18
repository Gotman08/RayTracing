/**
 * @file test_sphere.cpp
 * @brief Tests unitaires pour l'intersection rayon-sphere
 *
 * Ce fichier teste la methode hit() de la classe Sphere qui calcule
 * l'intersection entre un rayon et une sphere. On verifie les cas classiques :
 * intersection par le centre, rayon tangent, rayon qui rate la sphere,
 * rayon tire depuis l'interieur, sphere derriere le rayon, contraintes
 * d'intervalle de t. On verifie aussi les normales (face avant/arriere,
 * unitaire) et le comportement avec differents rayons et tailles de spheres.
 */

#include <iostream>
#include <cmath>

#include "raytracer/core/ray.cuh"
#include "raytracer/core/interval.cuh"
#include "raytracer/core/hit_record.cuh"
#include "raytracer/geometry/sphere.cuh"

using namespace rt;

/**
 * @brief Macro d'assertion simple pour les tests
 *
 * Verifie qu'une expression booleenne est vraie. Affiche un message
 * d'erreur avec le fichier et la ligne en cas d'echec.
 */
#define TEST_ASSERT(expr) \
    do { \
        if (!(expr)) { \
            std::cerr << "    FAILED: " << #expr << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            return false; \
        } \
    } while(0)

/**
 * @brief Macro d'assertion avec tolerance pour les comparaisons flottantes
 *
 * Compare deux valeurs flottantes avec une tolerance eps.
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
 * @brief Macro d'execution d'un test unitaire
 *
 * Lance un test, affiche son resultat et incremente les compteurs.
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

constexpr float EPS = 1e-5f;


/**
 * @brief Teste l'intersection d'un rayon passant par le centre de la sphere
 *
 * Un rayon tire depuis z=5 vers z=-1 sur une sphere unitaire centree
 * a l'origine doit toucher a t=4 (point d'impact a z=1).
 */
bool test_sphere_ray_through_center() {
    Sphere sphere(Point3(0.0f, 0.0f, 0.0f), 1.0f, nullptr);

    Ray ray(Point3(0.0f, 0.0f, 5.0f), Vec3(0.0f, 0.0f, -1.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = sphere.hit(ray, ray_t, rec);
    TEST_ASSERT(hit == true);

    TEST_ASSERT_NEAR(rec.t, 4.0f, EPS);
    TEST_ASSERT_NEAR(rec.p.z, 1.0f, EPS);

    return true;
}

/**
 * @brief Teste un rayon qui rate la sphere
 *
 * Un rayon decale de 2 unites sur l'axe X ne doit pas toucher
 * une sphere unitaire centree a l'origine.
 */
bool test_sphere_ray_misses() {
    Sphere sphere(Point3(0.0f, 0.0f, 0.0f), 1.0f, nullptr);

    Ray ray(Point3(2.0f, 0.0f, 5.0f), Vec3(0.0f, 0.0f, -1.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = sphere.hit(ray, ray_t, rec);
    TEST_ASSERT(hit == false);

    return true;
}

/**
 * @brief Teste un rayon tangent a la sphere
 *
 * Un rayon passant a exactement 1 unite du centre (x=1) doit toucher
 * la sphere en un seul point (cas tangent, discriminant = 0).
 */
bool test_sphere_ray_tangent() {
    Sphere sphere(Point3(0.0f, 0.0f, 0.0f), 1.0f, nullptr);

    Ray ray(Point3(1.0f, 0.0f, 5.0f), Vec3(0.0f, 0.0f, -1.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = sphere.hit(ray, ray_t, rec);
    TEST_ASSERT(hit == true);

    TEST_ASSERT_NEAR(rec.p.x, 1.0f, EPS);
    TEST_ASSERT_NEAR(rec.p.z, 0.0f, EPS);

    return true;
}

/**
 * @brief Teste un rayon tire depuis l'interieur de la sphere
 *
 * Quand l'origine du rayon est au centre de la sphere,
 * seule la racine positive (sortie de la sphere) doit etre retenue.
 * Le point d'impact doit etre a z=2 (rayon de 2).
 */
bool test_sphere_ray_from_inside() {
    Sphere sphere(Point3(0.0f, 0.0f, 0.0f), 2.0f, nullptr);

    Ray ray(Point3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, 1.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = sphere.hit(ray, ray_t, rec);
    TEST_ASSERT(hit == true);

    TEST_ASSERT_NEAR(rec.t, 2.0f, EPS);
    TEST_ASSERT_NEAR(rec.p.z, 2.0f, EPS);

    return true;
}

/**
 * @brief Teste un rayon dont la sphere est derriere lui
 *
 * Le rayon part de z=5 et va vers z positif, donc la sphere
 * a l'origine se trouve derriere. Aucune intersection valide.
 */
bool test_sphere_ray_behind() {
    Sphere sphere(Point3(0.0f, 0.0f, 0.0f), 1.0f, nullptr);

    Ray ray(Point3(0.0f, 0.0f, 5.0f), Vec3(0.0f, 0.0f, 1.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = sphere.hit(ray, ray_t, rec);
    TEST_ASSERT(hit == false);

    return true;
}

/**
 * @brief Teste qu'un intervalle de t trop restrictif exclut l'intersection
 *
 * Le rayon touche la sphere a t=4, mais l'intervalle autorise est
 * [0.001, 3.0], donc l'intersection doit etre rejetee.
 */
bool test_sphere_ray_interval_excludes() {
    Sphere sphere(Point3(0.0f, 0.0f, 0.0f), 1.0f, nullptr);

    Ray ray(Point3(0.0f, 0.0f, 5.0f), Vec3(0.0f, 0.0f, -1.0f));
    Interval ray_t(0.001f, 3.0f);
    HitRecord rec;

    bool hit = sphere.hit(ray, ray_t, rec);
    TEST_ASSERT(hit == false);

    return true;
}


/**
 * @brief Teste la normale a la surface pour un impact face avant
 *
 * Un rayon arrivant de face sur la sphere doit produire une normale
 * pointant vers le rayon (z=1) et front_face = true.
 */
bool test_sphere_normal_at_front() {
    Sphere sphere(Point3(0.0f, 0.0f, 0.0f), 1.0f, nullptr);

    Ray ray(Point3(0.0f, 0.0f, 5.0f), Vec3(0.0f, 0.0f, -1.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    sphere.hit(ray, ray_t, rec);

    TEST_ASSERT_NEAR(rec.normal.x, 0.0f, EPS);
    TEST_ASSERT_NEAR(rec.normal.y, 0.0f, EPS);
    TEST_ASSERT_NEAR(rec.normal.z, 1.0f, EPS);
    TEST_ASSERT(rec.front_face == true);

    return true;
}

/**
 * @brief Teste la normale pour un rayon tire depuis l'interieur
 *
 * La normale doit etre inversee (z=-1) et front_face = false
 * car le rayon touche la face interieure de la sphere.
 */
bool test_sphere_normal_from_inside() {
    Sphere sphere(Point3(0.0f, 0.0f, 0.0f), 2.0f, nullptr);

    Ray ray(Point3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, 1.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    sphere.hit(ray, ray_t, rec);

    TEST_ASSERT_NEAR(rec.normal.z, -1.0f, EPS);
    TEST_ASSERT(rec.front_face == false);

    return true;
}

/**
 * @brief Teste que la normale a la sphere est toujours unitaire
 *
 * Quel que soit le point d'impact et le rayon de la sphere,
 * la normale retournee doit avoir une longueur de 1.
 */
bool test_sphere_normal_is_unit() {
    Sphere sphere(Point3(1.0f, 2.0f, 3.0f), 2.5f, nullptr);

    Ray ray(Point3(10.0f, 10.0f, 10.0f), Vec3(-1.0f, -1.0f, -1.0f).normalized());
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = sphere.hit(ray, ray_t, rec);
    if (!hit) return false;

    float len = rec.normal.length();
    TEST_ASSERT_NEAR(len, 1.0f, EPS);

    return true;
}


/**
 * @brief Teste l'intersection avec une sphere dont le centre est decale
 *
 * La sphere est centree en (5, 0, 0) au lieu de l'origine.
 * Le rayon doit la toucher correctement.
 */
bool test_sphere_offset_center() {
    Sphere sphere(Point3(5.0f, 0.0f, 0.0f), 1.0f, nullptr);

    Ray ray(Point3(5.0f, 0.0f, 5.0f), Vec3(0.0f, 0.0f, -1.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = sphere.hit(ray, ray_t, rec);
    TEST_ASSERT(hit == true);
    TEST_ASSERT_NEAR(rec.p.x, 5.0f, EPS);
    TEST_ASSERT_NEAR(rec.p.z, 1.0f, EPS);

    return true;
}

/**
 * @brief Teste l'intersection avec une sphere de grand rayon (100)
 *
 * Verifie que le calcul reste precis pour de grandes spheres.
 * Le rayon depuis z=200 doit toucher a t=100.
 */
bool test_sphere_large_radius() {
    Sphere sphere(Point3(0.0f, 0.0f, 0.0f), 100.0f, nullptr);

    Ray ray(Point3(0.0f, 0.0f, 200.0f), Vec3(0.0f, 0.0f, -1.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = sphere.hit(ray, ray_t, rec);
    TEST_ASSERT(hit == true);
    TEST_ASSERT_NEAR(rec.t, 100.0f, EPS);

    return true;
}

/**
 * @brief Teste l'intersection avec une sphere de petit rayon (0.01)
 *
 * Verifie que le calcul reste precis meme pour de tres petites spheres.
 */
bool test_sphere_small_radius() {
    Sphere sphere(Point3(0.0f, 0.0f, 0.0f), 0.01f, nullptr);

    Ray ray(Point3(0.0f, 0.0f, 1.0f), Vec3(0.0f, 0.0f, -1.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = sphere.hit(ray, ray_t, rec);
    TEST_ASSERT(hit == true);
    TEST_ASSERT_NEAR(rec.p.z, 0.01f, EPS);

    return true;
}


/**
 * @brief Execute l'ensemble des tests unitaires pour Sphere
 *
 * Lance tous les tests d'intersection rayon-sphere et met a jour les compteurs.
 *
 * @param passed Compteur de tests reussis
 * @param failed Compteur de tests echoues
 * @param total Compteur du nombre total de tests executes
 */
void run_sphere_tests(int& passed, int& failed, int& total) {
    RUN_TEST(test_sphere_ray_through_center);
    RUN_TEST(test_sphere_ray_misses);
    RUN_TEST(test_sphere_ray_tangent);
    RUN_TEST(test_sphere_ray_from_inside);
    RUN_TEST(test_sphere_ray_behind);
    RUN_TEST(test_sphere_ray_interval_excludes);

    RUN_TEST(test_sphere_normal_at_front);
    RUN_TEST(test_sphere_normal_from_inside);
    RUN_TEST(test_sphere_normal_is_unit);

    RUN_TEST(test_sphere_offset_center);
    RUN_TEST(test_sphere_large_radius);
    RUN_TEST(test_sphere_small_radius);
}
