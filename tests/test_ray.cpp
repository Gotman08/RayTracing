/**
 * @file test_ray.cpp
 * @brief Tests unitaires pour la classe Ray (rayon)
 *
 * Ce fichier verifie le bon fonctionnement de la classe Ray qui represente
 * un rayon dans l'espace 3D, defini par une origine et une direction.
 * On teste les constructeurs (par defaut, parametre, avec temps),
 * les accesseurs (origin, direction, time) et la methode at(t)
 * qui calcule le point le long du rayon a la distance parametrique t.
 */

#include <iostream>
#include <cmath>

#include "raytracer/core/ray.cuh"

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
 * Compare deux valeurs flottantes avec une tolerance eps pour gerer
 * les imprecisions inherentes au calcul en virgule flottante.
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

constexpr float EPS = 1e-6f;


/**
 * @brief Teste le constructeur par defaut de Ray
 *
 * Verifie que le constructeur sans argument initialise l'origine
 * et la direction a (0, 0, 0) et le temps a 0.
 */
bool test_ray_default_constructor() {
    Ray r;
    TEST_ASSERT_NEAR(r.origin().x, 0.0f, EPS);
    TEST_ASSERT_NEAR(r.origin().y, 0.0f, EPS);
    TEST_ASSERT_NEAR(r.origin().z, 0.0f, EPS);
    TEST_ASSERT_NEAR(r.direction().x, 0.0f, EPS);
    TEST_ASSERT_NEAR(r.direction().y, 0.0f, EPS);
    TEST_ASSERT_NEAR(r.direction().z, 0.0f, EPS);
    TEST_ASSERT_NEAR(r.time(), 0.0f, EPS);
    return true;
}

/**
 * @brief Teste le constructeur parametre de Ray (origine + direction)
 *
 * Verifie que l'origine et la direction sont correctement stockees
 * et que le temps est initialise a 0 par defaut.
 */
bool test_ray_parameterized_constructor() {
    Point3 origin(1.0f, 2.0f, 3.0f);
    Vec3 direction(0.0f, 0.0f, -1.0f);
    Ray r(origin, direction);

    TEST_ASSERT_NEAR(r.origin().x, 1.0f, EPS);
    TEST_ASSERT_NEAR(r.origin().y, 2.0f, EPS);
    TEST_ASSERT_NEAR(r.origin().z, 3.0f, EPS);
    TEST_ASSERT_NEAR(r.direction().x, 0.0f, EPS);
    TEST_ASSERT_NEAR(r.direction().y, 0.0f, EPS);
    TEST_ASSERT_NEAR(r.direction().z, -1.0f, EPS);
    TEST_ASSERT_NEAR(r.time(), 0.0f, EPS);
    return true;
}

/**
 * @brief Teste le constructeur de Ray avec un parametre de temps
 *
 * Verifie que le temps est correctement pris en compte lors de
 * la construction du rayon. Utile pour le flou de mouvement (motion blur).
 */
bool test_ray_with_time() {
    Point3 origin(0.0f, 0.0f, 0.0f);
    Vec3 direction(1.0f, 0.0f, 0.0f);
    Ray r(origin, direction, 0.5f);

    TEST_ASSERT_NEAR(r.time(), 0.5f, EPS);
    return true;
}


/**
 * @brief Teste at(0) qui doit retourner l'origine du rayon
 *
 * Quand t=0, le point sur le rayon correspond exactement a l'origine.
 */
bool test_ray_at_zero() {
    Point3 origin(1.0f, 2.0f, 3.0f);
    Vec3 direction(1.0f, 0.0f, 0.0f);
    Ray r(origin, direction);

    Point3 p = r.at(0.0f);
    TEST_ASSERT_NEAR(p.x, 1.0f, EPS);
    TEST_ASSERT_NEAR(p.y, 2.0f, EPS);
    TEST_ASSERT_NEAR(p.z, 3.0f, EPS);
    return true;
}

/**
 * @brief Teste at() avec un parametre t positif
 *
 * Verifie que P(t) = origin + t * direction pour t=2.
 */
bool test_ray_at_positive() {
    Point3 origin(0.0f, 0.0f, 0.0f);
    Vec3 direction(1.0f, 2.0f, 3.0f);
    Ray r(origin, direction);

    Point3 p = r.at(2.0f);
    TEST_ASSERT_NEAR(p.x, 2.0f, EPS);
    TEST_ASSERT_NEAR(p.y, 4.0f, EPS);
    TEST_ASSERT_NEAR(p.z, 6.0f, EPS);
    return true;
}

/**
 * @brief Teste at() avec un parametre t negatif
 *
 * Un t negatif correspond a un point situe derriere l'origine
 * du rayon, dans la direction opposee.
 */
bool test_ray_at_negative() {
    Point3 origin(5.0f, 5.0f, 5.0f);
    Vec3 direction(1.0f, 0.0f, 0.0f);
    Ray r(origin, direction);

    Point3 p = r.at(-3.0f);
    TEST_ASSERT_NEAR(p.x, 2.0f, EPS);
    TEST_ASSERT_NEAR(p.y, 5.0f, EPS);
    TEST_ASSERT_NEAR(p.z, 5.0f, EPS);
    return true;
}

/**
 * @brief Teste at() avec un parametre t fractionnaire (0.5)
 *
 * Verifie le calcul pour un point situe a mi-chemin le long du rayon.
 */
bool test_ray_at_fractional() {
    Point3 origin(0.0f, 0.0f, 0.0f);
    Vec3 direction(4.0f, 0.0f, 0.0f);
    Ray r(origin, direction);

    Point3 p = r.at(0.5f);
    TEST_ASSERT_NEAR(p.x, 2.0f, EPS);
    TEST_ASSERT_NEAR(p.y, 0.0f, EPS);
    TEST_ASSERT_NEAR(p.z, 0.0f, EPS);
    return true;
}

/**
 * @brief Teste at(1) avec une direction normalisee
 *
 * Avec une direction unitaire, at(1) doit donner un point
 * a exactement 1 unite de distance de l'origine.
 */
bool test_ray_at_unit_direction() {
    Point3 origin(0.0f, 0.0f, 0.0f);
    Vec3 direction = Vec3(1.0f, 1.0f, 1.0f).normalized();
    Ray r(origin, direction);

    Point3 p = r.at(1.0f);
    float dist = (p - origin).length();
    TEST_ASSERT_NEAR(dist, 1.0f, EPS);
    return true;
}


/**
 * @brief Teste un rayon avec une direction diagonale (1, 1, 1)
 *
 * Verifie le calcul de at() pour un rayon qui se deplace
 * dans les trois dimensions simultanement.
 */
bool test_ray_diagonal_direction() {
    Point3 origin(1.0f, 1.0f, 1.0f);
    Vec3 direction(1.0f, 1.0f, 1.0f);
    Ray r(origin, direction);

    Point3 p = r.at(3.0f);
    TEST_ASSERT_NEAR(p.x, 4.0f, EPS);
    TEST_ASSERT_NEAR(p.y, 4.0f, EPS);
    TEST_ASSERT_NEAR(p.z, 4.0f, EPS);
    return true;
}

/**
 * @brief Teste un rayon oriente le long de l'axe -Z
 *
 * Configuration typique en raytracing ou la camera regarde
 * vers les Z negatifs. Verifie at(10) sur cet axe.
 */
bool test_ray_z_direction() {
    Point3 origin(0.0f, 0.0f, 0.0f);
    Vec3 direction(0.0f, 0.0f, -1.0f);
    Ray r(origin, direction);

    Point3 p = r.at(10.0f);
    TEST_ASSERT_NEAR(p.x, 0.0f, EPS);
    TEST_ASSERT_NEAR(p.y, 0.0f, EPS);
    TEST_ASSERT_NEAR(p.z, -10.0f, EPS);
    return true;
}


/**
 * @brief Execute l'ensemble des tests unitaires pour Ray
 *
 * Lance tous les tests de la classe Ray et met a jour les compteurs.
 *
 * @param passed Compteur de tests reussis
 * @param failed Compteur de tests echoues
 * @param total Compteur du nombre total de tests executes
 */
void run_ray_tests(int& passed, int& failed, int& total) {
    RUN_TEST(test_ray_default_constructor);
    RUN_TEST(test_ray_parameterized_constructor);
    RUN_TEST(test_ray_with_time);

    RUN_TEST(test_ray_at_zero);
    RUN_TEST(test_ray_at_positive);
    RUN_TEST(test_ray_at_negative);
    RUN_TEST(test_ray_at_fractional);
    RUN_TEST(test_ray_at_unit_direction);

    RUN_TEST(test_ray_diagonal_direction);
    RUN_TEST(test_ray_z_direction);
}
