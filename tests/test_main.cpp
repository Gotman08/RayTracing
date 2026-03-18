/**
 * @file test_main.cpp
 * @brief Point d'entree principal de la suite de tests unitaires du raytracer.
 *
 * Ce fichier contient la fonction main() qui orchestre l'execution de toutes
 * les suites de tests : Vec3, Ray, Interval, Sphere, Plane, AABB, Materials,
 * Camera, Tone Mapping et BVH. Il affiche un recapitulatif des resultats
 * a la fin et retourne un code d'erreur si des tests ont echoue.
 */

#include <iostream>
#include <cmath>
#include <string>


/**
 * @brief Macro d'assertion simple pour les tests.
 *
 * Verifie qu'une expression est vraie, sinon affiche un message d'erreur
 * avec le fichier et la ligne, puis retourne false.
 */
#define TEST_ASSERT(expr) \
    do { \
        if (!(expr)) { \
            std::cerr << "    FAILED: " << #expr << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            return false; \
        } \
    } while(0)

/**
 * @brief Macro d'assertion avec message personnalise.
 *
 * Comme TEST_ASSERT, mais affiche un message explicatif fourni par l'utilisateur
 * en cas d'echec, au lieu de l'expression brute.
 */
#define TEST_ASSERT_MSG(expr, msg) \
    do { \
        if (!(expr)) { \
            std::cerr << "    FAILED: " << msg << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            return false; \
        } \
    } while(0)

/**
 * @brief Macro d'assertion avec tolerance pour comparer des valeurs flottantes.
 *
 * Verifie que |a - b| <= eps. Affiche les valeurs et l'epsilon en cas d'echec.
 */
#define TEST_ASSERT_NEAR(a, b, eps) \
    do { \
        if (std::abs((a) - (b)) > (eps)) { \
            std::cerr << "    FAILED: " << #a << " (" << (a) << ") != " << #b << " (" << (b) << ")" \
                      << " (eps=" << (eps) << ") at " << __FILE__ << ":" << __LINE__ << "\n"; \
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


/** @brief Lance les tests unitaires du type Vec3 (vecteur 3D). */
void run_vec3_tests(int& passed, int& failed, int& total);
/** @brief Lance les tests unitaires du type Ray (rayon). */
void run_ray_tests(int& passed, int& failed, int& total);
/** @brief Lance les tests unitaires du type Interval. */
void run_interval_tests(int& passed, int& failed, int& total);
/** @brief Lance les tests unitaires de la geometrie Sphere. */
void run_sphere_tests(int& passed, int& failed, int& total);
/** @brief Lance les tests unitaires de la geometrie Plane (plan). */
void run_plane_tests(int& passed, int& failed, int& total);
/** @brief Lance les tests unitaires de l'AABB (boite englobante alignee aux axes). */
void run_aabb_tests(int& passed, int& failed, int& total);
/** @brief Lance les tests unitaires des materiaux (lambertien, metal, dielectrique). */
void run_materials_tests(int& passed, int& failed, int& total);
/** @brief Lance les tests unitaires de la camera (initialisation, generation de rayons). */
void run_camera_tests(int& passed, int& failed, int& total);
/** @brief Lance les tests unitaires du tone mapping Reinhard et de la correction gamma. */
void run_tone_mapping_tests(int& passed, int& failed, int& total);
/** @brief Lance les tests unitaires du BVH (construction, traversee, intersection). */
void run_bvh_tests(int& passed, int& failed, int& total);


/**
 * @brief Fonction principale : execute toutes les suites de tests et affiche le bilan.
 *
 * Chaque suite de tests est lancee l'une apres l'autre. A la fin, un recapitulatif
 * indique le nombre de tests passes et echoues. Le programme retourne 0 si tous
 * les tests ont reussi, ou 1 si au moins un test a echoue.
 *
 * @return 0 si tous les tests passent, 1 sinon.
 */
int main() {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "   CUDA Ray Tracer - Unit Tests\n";
    std::cout << "========================================\n\n";

    int total_passed = 0;
    int total_failed = 0;
    int total_tests = 0;

    std::cout << "[Vec3 Tests]\n";
    run_vec3_tests(total_passed, total_failed, total_tests);
    std::cout << "\n";

    std::cout << "[Ray Tests]\n";
    run_ray_tests(total_passed, total_failed, total_tests);
    std::cout << "\n";

    std::cout << "[Interval Tests]\n";
    run_interval_tests(total_passed, total_failed, total_tests);
    std::cout << "\n";

    std::cout << "[Sphere Tests]\n";
    run_sphere_tests(total_passed, total_failed, total_tests);
    std::cout << "\n";

    std::cout << "[Plane Tests]\n";
    run_plane_tests(total_passed, total_failed, total_tests);
    std::cout << "\n";

    std::cout << "[AABB Tests]\n";
    run_aabb_tests(total_passed, total_failed, total_tests);
    std::cout << "\n";

    std::cout << "[Materials Tests]\n";
    run_materials_tests(total_passed, total_failed, total_tests);
    std::cout << "\n";

    std::cout << "[Camera Tests]\n";
    run_camera_tests(total_passed, total_failed, total_tests);
    std::cout << "\n";

    std::cout << "[Tone Mapping Tests]\n";
    run_tone_mapping_tests(total_passed, total_failed, total_tests);
    std::cout << "\n";

    std::cout << "[BVH Tests]\n";
    run_bvh_tests(total_passed, total_failed, total_tests);
    std::cout << "\n";

    std::cout << "========================================\n";
    std::cout << "   Results: " << total_passed << "/" << total_tests << " passed";
    if (total_failed > 0) {
        std::cout << " (" << total_failed << " failed)";
    }
    std::cout << "\n";
    std::cout << "========================================\n\n";

    return (total_failed > 0) ? 1 : 0;
}
