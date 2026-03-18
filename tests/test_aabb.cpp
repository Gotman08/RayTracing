/**
 * @file test_aabb.cpp
 * @brief Tests unitaires pour la classe AABB (Axis-Aligned Bounding Box)
 *
 * Ce fichier teste la boite englobante alignee sur les axes, structure
 * fondamentale pour l'acceleration du raytracing via les BVH (Bounding
 * Volume Hierarchy). On verifie les differents constructeurs (points,
 * intervalles, fusion), les tests d'intersection rayon-boite,
 * le calcul du centroide, de l'aire de surface, de l'axe le plus long
 * et l'acces aux intervalles par axe.
 */

#include <iostream>
#include <cmath>

#include "raytracer/core/aabb.cuh"
#include "raytracer/core/cuda_utils.cuh"

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

constexpr float EPS = 1e-4f;


/**
 * @brief Teste le constructeur AABB a partir de deux points
 *
 * Verifie que la boite englobante est correctement construite
 * avec les intervalles [0,2], [0,3], [0,4] sur chaque axe.
 */
bool test_aabb_point_constructor() {
    Point3 a(0.0f, 0.0f, 0.0f);
    Point3 b(2.0f, 3.0f, 4.0f);
    AABB box(a, b);

    TEST_ASSERT_NEAR(box.x.min, 0.0f, EPS);
    TEST_ASSERT_NEAR(box.x.max, 2.0f, EPS);
    TEST_ASSERT_NEAR(box.y.min, 0.0f, EPS);
    TEST_ASSERT_NEAR(box.y.max, 3.0f, EPS);
    TEST_ASSERT_NEAR(box.z.min, 0.0f, EPS);
    TEST_ASSERT_NEAR(box.z.max, 4.0f, EPS);

    return true;
}

/**
 * @brief Teste le constructeur avec des points dans l'ordre inverse
 *
 * Verifie que le constructeur gere le cas ou le premier point a des
 * coordonnees plus grandes que le second (les bornes doivent etre triees).
 */
bool test_aabb_point_constructor_reversed() {
    Point3 a(5.0f, 5.0f, 5.0f);
    Point3 b(0.0f, 0.0f, 0.0f);
    AABB box(a, b);

    TEST_ASSERT_NEAR(box.x.min, 0.0f, EPS);
    TEST_ASSERT_NEAR(box.x.max, 5.0f, EPS);

    return true;
}

/**
 * @brief Teste le constructeur a partir de trois intervalles (un par axe)
 *
 * Verifie que la boite est correctement initialisee lorsqu'on
 * fournit directement les intervalles pour X, Y et Z.
 */
bool test_aabb_interval_constructor() {
    Interval ix(0.0f, 2.0f);
    Interval iy(0.0f, 3.0f);
    Interval iz(0.0f, 4.0f);
    AABB box(ix, iy, iz);

    TEST_ASSERT_NEAR(box.x.min, 0.0f, EPS);
    TEST_ASSERT_NEAR(box.x.max, 2.0f, EPS);
    TEST_ASSERT_NEAR(box.y.min, 0.0f, EPS);
    TEST_ASSERT_NEAR(box.y.max, 3.0f, EPS);
    TEST_ASSERT_NEAR(box.z.min, 0.0f, EPS);
    TEST_ASSERT_NEAR(box.z.max, 4.0f, EPS);

    return true;
}

/**
 * @brief Teste le constructeur de fusion de deux AABB
 *
 * La fusion de deux boites disjointes doit produire une boite
 * englobant les deux. Utilise dans la construction du BVH.
 */
bool test_aabb_merge_constructor() {
    AABB box1(Point3(0.0f, 0.0f, 0.0f), Point3(1.0f, 1.0f, 1.0f));
    AABB box2(Point3(2.0f, 2.0f, 2.0f), Point3(3.0f, 3.0f, 3.0f));
    AABB merged(box1, box2);

    TEST_ASSERT_NEAR(merged.x.min, 0.0f, EPS);
    TEST_ASSERT_NEAR(merged.x.max, 3.0f, EPS);
    TEST_ASSERT_NEAR(merged.y.min, 0.0f, EPS);
    TEST_ASSERT_NEAR(merged.y.max, 3.0f, EPS);

    return true;
}


/**
 * @brief Teste l'intersection d'un rayon passant par le centre de la boite
 *
 * Un rayon tire le long de l'axe Z vers le centre de la boite
 * doit produire une intersection.
 */
bool test_aabb_hit_through_center() {
    AABB box(Point3(0.0f, 0.0f, 0.0f), Point3(2.0f, 2.0f, 2.0f));

    Ray ray(Point3(1.0f, 1.0f, -5.0f), Vec3(0.0f, 0.0f, 1.0f));
    Interval ray_t(0.001f, INFINITY_F);

    bool hit = box.hit(ray, ray_t);
    TEST_ASSERT(hit == true);

    return true;
}

/**
 * @brief Teste l'intersection d'un rayon rasant le bord de la boite
 *
 * Un rayon passant pres du coin de la boite doit quand meme
 * etre detecte comme intersection.
 */
bool test_aabb_hit_edge() {
    AABB box(Point3(0.0f, 0.0f, 0.0f), Point3(2.0f, 2.0f, 2.0f));

    Ray ray(Point3(0.001f, 0.001f, -5.0f), Vec3(0.0f, 0.0f, 1.0f));
    Interval ray_t(0.001f, INFINITY_F);

    bool hit = box.hit(ray, ray_t);
    TEST_ASSERT(hit == true);

    return true;
}

/**
 * @brief Teste un rayon qui rate completement la boite
 *
 * Un rayon tire loin de la boite ne doit pas produire d'intersection.
 */
bool test_aabb_miss() {
    AABB box(Point3(0.0f, 0.0f, 0.0f), Point3(2.0f, 2.0f, 2.0f));

    Ray ray(Point3(5.0f, 5.0f, -5.0f), Vec3(0.0f, 0.0f, 1.0f));
    Interval ray_t(0.001f, INFINITY_F);

    bool hit = box.hit(ray, ray_t);
    TEST_ASSERT(hit == false);

    return true;
}

/**
 * @brief Teste un rayon dont la direction s'eloigne de la boite
 *
 * Le rayon pointe dans la direction opposee a la boite,
 * donc aucune intersection ne doit etre trouvee.
 */
bool test_aabb_ray_behind() {
    AABB box(Point3(0.0f, 0.0f, 0.0f), Point3(2.0f, 2.0f, 2.0f));

    Ray ray(Point3(1.0f, 1.0f, -5.0f), Vec3(0.0f, 0.0f, -1.0f));
    Interval ray_t(0.001f, INFINITY_F);

    bool hit = box.hit(ray, ray_t);
    TEST_ASSERT(hit == false);

    return true;
}

/**
 * @brief Teste un rayon dont l'origine est a l'interieur de la boite
 *
 * Un rayon partant de l'interieur de la boite doit toujours
 * detecter une intersection (en sortie de boite).
 */
bool test_aabb_ray_inside() {
    AABB box(Point3(0.0f, 0.0f, 0.0f), Point3(10.0f, 10.0f, 10.0f));

    Ray ray(Point3(5.0f, 5.0f, 5.0f), Vec3(1.0f, 0.0f, 0.0f));
    Interval ray_t(0.001f, INFINITY_F);

    bool hit = box.hit(ray, ray_t);
    TEST_ASSERT(hit == true);

    return true;
}

/**
 * @brief Teste un rayon diagonal traversant la boite
 *
 * Un rayon en diagonale depuis (-1,-1,-1) vers (1,1,1) doit
 * traverser la boite [0,2]^3.
 */
bool test_aabb_diagonal_ray() {
    AABB box(Point3(0.0f, 0.0f, 0.0f), Point3(2.0f, 2.0f, 2.0f));

    Ray ray(Point3(-1.0f, -1.0f, -1.0f), Vec3(1.0f, 1.0f, 1.0f).normalized());
    Interval ray_t(0.001f, INFINITY_F);

    bool hit = box.hit(ray, ray_t);
    TEST_ASSERT(hit == true);

    return true;
}

/**
 * @brief Teste qu'un intervalle de t trop restrictif exclut l'intersection
 *
 * Meme si le rayon touche la boite geometriquement, si le t d'intersection
 * est en dehors de l'intervalle [tmin, tmax], le hit doit echouer.
 */
bool test_aabb_interval_excludes() {
    AABB box(Point3(0.0f, 0.0f, 0.0f), Point3(2.0f, 2.0f, 2.0f));

    Ray ray(Point3(1.0f, 1.0f, -10.0f), Vec3(0.0f, 0.0f, 1.0f));
    Interval ray_t(0.001f, 5.0f);

    bool hit = box.hit(ray, ray_t);
    TEST_ASSERT(hit == false);

    return true;
}


/**
 * @brief Teste le calcul du centroide de la boite
 *
 * Le centroide de [0,4] x [0,6] x [0,8] doit etre (2, 3, 4).
 * Le centroide est utilise pour le tri spatial dans le BVH.
 */
bool test_aabb_centroid() {
    AABB box(Point3(0.0f, 0.0f, 0.0f), Point3(4.0f, 6.0f, 8.0f));
    Point3 c = box.centroid();

    TEST_ASSERT_NEAR(c.x, 2.0f, EPS);
    TEST_ASSERT_NEAR(c.y, 3.0f, EPS);
    TEST_ASSERT_NEAR(c.z, 4.0f, EPS);

    return true;
}

/**
 * @brief Teste le centroide d'une boite decalee par rapport a l'origine
 *
 * Le centroide de [2,4]^3 doit etre (3, 3, 3).
 */
bool test_aabb_centroid_offset() {
    AABB box(Point3(2.0f, 2.0f, 2.0f), Point3(4.0f, 4.0f, 4.0f));
    Point3 c = box.centroid();

    TEST_ASSERT_NEAR(c.x, 3.0f, EPS);
    TEST_ASSERT_NEAR(c.y, 3.0f, EPS);
    TEST_ASSERT_NEAR(c.z, 3.0f, EPS);

    return true;
}


/**
 * @brief Teste l'aire de surface d'un cube
 *
 * Un cube de cote 2 a une aire de surface de 6 * 2^2 = 24.
 * L'aire de surface est utilisee dans l'heuristique SAH du BVH.
 */
bool test_aabb_surface_area_cube() {
    AABB box(Point3(0.0f, 0.0f, 0.0f), Point3(2.0f, 2.0f, 2.0f));
    float sa = box.surface_area();

    TEST_ASSERT_NEAR(sa, 24.0f, EPS);

    return true;
}

/**
 * @brief Teste l'aire de surface d'un parallelepipede rectangle
 *
 * Un parallelepipede 2x3x4 a une aire de 2*(2*3 + 2*4 + 3*4) = 52.
 */
bool test_aabb_surface_area_rect() {
    AABB box(Point3(0.0f, 0.0f, 0.0f), Point3(2.0f, 3.0f, 4.0f));
    float sa = box.surface_area();

    TEST_ASSERT_NEAR(sa, 52.0f, EPS);

    return true;
}


/**
 * @brief Teste la detection de l'axe le plus long (axe X)
 *
 * Pour une boite 10x2x3, l'axe X (indice 0) est le plus long.
 * Utilise pour choisir l'axe de partition dans le BVH.
 */
bool test_aabb_longest_axis_x() {
    AABB box(Point3(0.0f, 0.0f, 0.0f), Point3(10.0f, 2.0f, 3.0f));
    TEST_ASSERT(box.longest_axis() == 0);

    return true;
}

/**
 * @brief Teste la detection de l'axe le plus long (axe Y)
 *
 * Pour une boite 2x10x3, l'axe Y (indice 1) est le plus long.
 */
bool test_aabb_longest_axis_y() {
    AABB box(Point3(0.0f, 0.0f, 0.0f), Point3(2.0f, 10.0f, 3.0f));
    TEST_ASSERT(box.longest_axis() == 1);

    return true;
}

/**
 * @brief Teste la detection de l'axe le plus long (axe Z)
 *
 * Pour une boite 2x3x10, l'axe Z (indice 2) est le plus long.
 */
bool test_aabb_longest_axis_z() {
    AABB box(Point3(0.0f, 0.0f, 0.0f), Point3(2.0f, 3.0f, 10.0f));
    TEST_ASSERT(box.longest_axis() == 2);

    return true;
}


/**
 * @brief Teste l'acces aux intervalles par indice d'axe
 *
 * Verifie que axis_interval(0), axis_interval(1) et axis_interval(2)
 * retournent respectivement les intervalles X, Y et Z de la boite.
 */
bool test_aabb_axis_interval() {
    AABB box(Point3(1.0f, 2.0f, 3.0f), Point3(4.0f, 5.0f, 6.0f));

    const Interval& ix = box.axis_interval(0);
    const Interval& iy = box.axis_interval(1);
    const Interval& iz = box.axis_interval(2);

    TEST_ASSERT_NEAR(ix.min, 1.0f, EPS);
    TEST_ASSERT_NEAR(ix.max, 4.0f, EPS);
    TEST_ASSERT_NEAR(iy.min, 2.0f, EPS);
    TEST_ASSERT_NEAR(iy.max, 5.0f, EPS);
    TEST_ASSERT_NEAR(iz.min, 3.0f, EPS);
    TEST_ASSERT_NEAR(iz.max, 6.0f, EPS);

    return true;
}


/**
 * @brief Execute l'ensemble des tests unitaires pour AABB
 *
 * Lance tous les tests de la classe AABB et met a jour les compteurs.
 *
 * @param passed Compteur de tests reussis
 * @param failed Compteur de tests echoues
 * @param total Compteur du nombre total de tests executes
 */
void run_aabb_tests(int& passed, int& failed, int& total) {
    RUN_TEST(test_aabb_point_constructor);
    RUN_TEST(test_aabb_point_constructor_reversed);
    RUN_TEST(test_aabb_interval_constructor);
    RUN_TEST(test_aabb_merge_constructor);

    RUN_TEST(test_aabb_hit_through_center);
    RUN_TEST(test_aabb_hit_edge);
    RUN_TEST(test_aabb_miss);
    RUN_TEST(test_aabb_ray_behind);
    RUN_TEST(test_aabb_ray_inside);
    RUN_TEST(test_aabb_diagonal_ray);
    RUN_TEST(test_aabb_interval_excludes);

    RUN_TEST(test_aabb_centroid);
    RUN_TEST(test_aabb_centroid_offset);

    RUN_TEST(test_aabb_surface_area_cube);
    RUN_TEST(test_aabb_surface_area_rect);

    RUN_TEST(test_aabb_longest_axis_x);
    RUN_TEST(test_aabb_longest_axis_y);
    RUN_TEST(test_aabb_longest_axis_z);

    RUN_TEST(test_aabb_axis_interval);
}
