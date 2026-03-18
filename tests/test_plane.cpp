/**
 * @file test_plane.cpp
 * @brief Tests unitaires pour l'intersection rayon-plan
 *
 * Ce fichier teste la methode hit() de la classe Plane qui calcule
 * l'intersection entre un rayon et un plan infini. On verifie les cas
 * classiques : intersection perpendiculaire, oblique, rayon parallele
 * au plan, rayon pointant dans la direction opposee, rayon venant
 * de derriere, contraintes d'intervalle de t. On teste aussi les
 * normales (face avant/arriere, unitaire) et des configurations
 * geometriques variees (plan vertical, plan diagonal, plan decale).
 */

#include <iostream>
#include <cmath>

#include "raytracer/core/ray.cuh"
#include "raytracer/core/interval.cuh"
#include "raytracer/core/hit_record.cuh"
#include "raytracer/geometry/plane.cuh"

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
 * @brief Teste l'intersection perpendiculaire d'un rayon avec un plan horizontal
 *
 * Un rayon vertical descendant depuis y=5 vers un plan horizontal (y=0)
 * doit toucher a t=5 au point y=0.
 */
bool test_plane_ray_perpendicular() {
    Plane plane(Point3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), nullptr);

    Ray ray(Point3(0.0f, 5.0f, 0.0f), Vec3(0.0f, -1.0f, 0.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = plane.hit(ray, ray_t, rec);
    TEST_ASSERT(hit == true);

    TEST_ASSERT_NEAR(rec.t, 5.0f, EPS);
    TEST_ASSERT_NEAR(rec.p.y, 0.0f, EPS);

    return true;
}

/**
 * @brief Teste l'intersection d'un rayon oblique avec un plan horizontal
 *
 * Un rayon en diagonale (direction normalisee (0,-1,-1)) doit toucher
 * le plan y=0 au point ou y vaut exactement 0.
 */
bool test_plane_ray_oblique() {
    Plane plane(Point3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), nullptr);

    Ray ray(Point3(0.0f, 5.0f, 5.0f), Vec3(0.0f, -1.0f, -1.0f).normalized());
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = plane.hit(ray, ray_t, rec);
    TEST_ASSERT(hit == true);

    TEST_ASSERT_NEAR(rec.p.y, 0.0f, EPS);

    return true;
}

/**
 * @brief Teste un rayon parallele au plan (aucune intersection)
 *
 * Un rayon se deplacant le long de l'axe X au dessus du plan horizontal
 * ne doit jamais le toucher car il est parallele.
 */
bool test_plane_ray_parallel() {
    Plane plane(Point3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), nullptr);

    Ray ray(Point3(0.0f, 5.0f, 0.0f), Vec3(1.0f, 0.0f, 0.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = plane.hit(ray, ray_t, rec);
    TEST_ASSERT(hit == false);

    return true;
}

/**
 * @brief Teste un rayon pointant dans la direction opposee au plan
 *
 * Le rayon est au dessus du plan et pointe vers le haut (y positif),
 * donc il s'eloigne du plan et ne doit pas l'intersecter.
 */
bool test_plane_ray_pointing_away() {
    Plane plane(Point3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), nullptr);

    Ray ray(Point3(0.0f, 5.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = plane.hit(ray, ray_t, rec);
    TEST_ASSERT(hit == false);

    return true;
}

/**
 * @brief Teste un rayon arrivant par l'arriere du plan
 *
 * Le rayon part de y=-5 et monte vers le plan. L'intersection
 * existe mais front_face doit etre false car le rayon touche
 * la face arriere du plan.
 */
bool test_plane_ray_from_behind() {
    Plane plane(Point3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), nullptr);

    Ray ray(Point3(0.0f, -5.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = plane.hit(ray, ray_t, rec);
    TEST_ASSERT(hit == true);

    TEST_ASSERT(rec.front_face == false);

    return true;
}

/**
 * @brief Teste qu'un intervalle de t trop restrictif exclut l'intersection
 *
 * Le rayon touche le plan a t=5, mais l'intervalle autorise est
 * [0.001, 3.0], donc l'intersection doit etre rejetee.
 */
bool test_plane_ray_interval_excludes() {
    Plane plane(Point3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), nullptr);

    Ray ray(Point3(0.0f, 5.0f, 0.0f), Vec3(0.0f, -1.0f, 0.0f));
    Interval ray_t(0.001f, 3.0f);
    HitRecord rec;

    bool hit = plane.hit(ray, ray_t, rec);
    TEST_ASSERT(hit == false);

    return true;
}


/**
 * @brief Teste la normale pour un impact sur la face avant du plan
 *
 * Le rayon arrive du cote de la normale, donc front_face = true
 * et la normale pointe vers le rayon (y=1).
 */
bool test_plane_normal_front_face() {
    Plane plane(Point3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), nullptr);

    Ray ray(Point3(0.0f, 5.0f, 0.0f), Vec3(0.0f, -1.0f, 0.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    plane.hit(ray, ray_t, rec);

    TEST_ASSERT_NEAR(rec.normal.y, 1.0f, EPS);
    TEST_ASSERT(rec.front_face == true);

    return true;
}

/**
 * @brief Teste la normale pour un impact sur la face arriere du plan
 *
 * Le rayon arrive du cote oppose a la normale, donc front_face = false
 * et la normale est inversee (y=-1) pour pointer vers le rayon.
 */
bool test_plane_normal_back_face() {
    Plane plane(Point3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), nullptr);

    Ray ray(Point3(0.0f, -5.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    plane.hit(ray, ray_t, rec);

    TEST_ASSERT_NEAR(rec.normal.y, -1.0f, EPS);
    TEST_ASSERT(rec.front_face == false);

    return true;
}

/**
 * @brief Teste que la normale au plan est toujours unitaire
 *
 * Meme si la normale fournie au constructeur n'est pas unitaire
 * (ici (0, 5, 0)), la normale dans le HitRecord doit avoir
 * une longueur de 1.
 */
bool test_plane_normal_is_unit() {
    Plane plane(Point3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 5.0f, 0.0f), nullptr);

    Ray ray(Point3(0.0f, 5.0f, 0.0f), Vec3(0.0f, -1.0f, 0.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    plane.hit(ray, ray_t, rec);

    float len = rec.normal.length();
    TEST_ASSERT_NEAR(len, 1.0f, EPS);

    return true;
}


/**
 * @brief Teste l'intersection avec un plan vertical (normal selon X)
 *
 * Un plan vertical en x=3 avec la normale (1,0,0) doit etre touche
 * par un rayon horizontal a t=3.
 */
bool test_plane_vertical_xz() {
    Plane plane(Point3(3.0f, 0.0f, 0.0f), Vec3(1.0f, 0.0f, 0.0f), nullptr);

    Ray ray(Point3(0.0f, 0.0f, 0.0f), Vec3(1.0f, 0.0f, 0.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = plane.hit(ray, ray_t, rec);
    TEST_ASSERT(hit == true);
    TEST_ASSERT_NEAR(rec.t, 3.0f, EPS);
    TEST_ASSERT_NEAR(rec.p.x, 3.0f, EPS);

    return true;
}

/**
 * @brief Teste l'intersection avec un plan diagonal (normale a 45 degres)
 *
 * Verifie que le point d'impact appartient bien au plan en testant
 * que le produit scalaire entre (p - point_du_plan) et la normale est nul.
 */
bool test_plane_diagonal() {
    Vec3 normal = Vec3(1.0f, 1.0f, 0.0f).normalized();
    Plane plane(Point3(5.0f, 5.0f, 0.0f), normal, nullptr);

    Ray ray(Point3(0.0f, 0.0f, 0.0f), Vec3(1.0f, 1.0f, 0.0f).normalized());
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = plane.hit(ray, ray_t, rec);
    TEST_ASSERT(hit == true);

    Vec3 diff = rec.p - Point3(5.0f, 5.0f, 0.0f);
    float check = dot(diff, normal);
    TEST_ASSERT_NEAR(check, 0.0f, EPS);

    return true;
}

/**
 * @brief Teste l'intersection avec un plan decale en hauteur (y=10)
 *
 * Le plan horizontal est a y=10 au lieu de y=0. Le rayon depuis y=15
 * doit toucher a t=5 au point y=10.
 */
bool test_plane_offset_point() {
    Plane plane(Point3(0.0f, 10.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), nullptr);

    Ray ray(Point3(0.0f, 15.0f, 0.0f), Vec3(0.0f, -1.0f, 0.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = plane.hit(ray, ray_t, rec);
    TEST_ASSERT(hit == true);
    TEST_ASSERT_NEAR(rec.p.y, 10.0f, EPS);
    TEST_ASSERT_NEAR(rec.t, 5.0f, EPS);

    return true;
}


/**
 * @brief Execute l'ensemble des tests unitaires pour Plane
 *
 * Lance tous les tests d'intersection rayon-plan et met a jour les compteurs.
 *
 * @param passed Compteur de tests reussis
 * @param failed Compteur de tests echoues
 * @param total Compteur du nombre total de tests executes
 */
void run_plane_tests(int& passed, int& failed, int& total) {
    RUN_TEST(test_plane_ray_perpendicular);
    RUN_TEST(test_plane_ray_oblique);
    RUN_TEST(test_plane_ray_parallel);
    RUN_TEST(test_plane_ray_pointing_away);
    RUN_TEST(test_plane_ray_from_behind);
    RUN_TEST(test_plane_ray_interval_excludes);

    RUN_TEST(test_plane_normal_front_face);
    RUN_TEST(test_plane_normal_back_face);
    RUN_TEST(test_plane_normal_is_unit);

    RUN_TEST(test_plane_vertical_xz);
    RUN_TEST(test_plane_diagonal);
    RUN_TEST(test_plane_offset_point);
}
