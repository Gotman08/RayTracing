/**
 * @file test_camera.cpp
 * @brief Tests unitaires pour la classe Camera du raytracer.
 *
 * Ce fichier contient l'ensemble des tests relatifs a la camera :
 * initialisation des parametres (centre, dimensions, systeme de coordonnees),
 * gestion du defocus blur (profondeur de champ), generation de rayons
 * depuis differents pixels, temps d'obturation et influence du champ de vision (FOV).
 */

#include <iostream>
#include <cmath>

#include "raytracer/core/vec3.cuh"
#include "raytracer/core/ray.cuh"
#include "raytracer/core/random.cuh"

#ifndef __CUDACC__
inline float curand_uniform(curandState*) { return 0.5f; }
namespace rt {
    inline Vec3 random_unit_vector(curandState*) { return Vec3(0, 1, 0); }
    inline Vec3 random_in_unit_sphere(curandState*) { return Vec3(0, 0, 0); }
    inline Vec3 random_in_unit_disk(curandState*) { return Vec3(0, 0, 0); }
}
#endif

#include "raytracer/camera/camera.cuh"

using namespace rt;

/**
 * @brief Macro d'assertion simple pour les tests.
 *
 * Verifie qu'une expression booleenne est vraie. Si elle est fausse,
 * affiche un message d'erreur avec le fichier et la ligne, puis retourne false
 * pour signaler l'echec du test.
 */
#define TEST_ASSERT(expr) \
    do { \
        if (!(expr)) { \
            std::cerr << "    FAILED: " << #expr << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            return false; \
        } \
    } while(0)

/**
 * @brief Macro d'assertion avec tolerance pour comparer des flottants.
 *
 * Verifie que la difference absolue entre deux valeurs est inferieure a un
 * epsilon donne. Utile pour les comparaisons de nombres a virgule flottante
 * ou les erreurs d'arrondi sont inevitables.
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
 * @brief Macro d'execution d'un test avec comptage des resultats.
 *
 * Execute une fonction de test et affiche [PASS] ou [FAIL] selon le resultat.
 * Met a jour les compteurs de tests passes, echoues et total.
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
 * @brief Verifie que le centre de la camera est correctement initialise a l'origine.
 */
static bool test_camera_center() {
    Camera cam;
    cam.initialize(800, 600, Point3(0, 0, 0), Point3(0, 0, -1), Vec3(0, 1, 0), 90.0f);
    TEST_ASSERT_NEAR(cam.center.x, 0.0f, 1e-5f);
    TEST_ASSERT_NEAR(cam.center.y, 0.0f, 1e-5f);
    TEST_ASSERT_NEAR(cam.center.z, 0.0f, 1e-5f);
    return true;
}

/**
 * @brief Verifie que les dimensions de l'image (largeur et hauteur) sont bien stockees.
 */
static bool test_camera_dimensions() {
    Camera cam;
    cam.initialize(800, 600, Point3(0, 0, 0), Point3(0, 0, -1), Vec3(0, 1, 0), 90.0f);
    TEST_ASSERT(cam.image_width == 800);
    TEST_ASSERT(cam.image_height == 600);
    return true;
}

/**
 * @brief Verifie que le repere local de la camera (u, v, w) est correctement calcule.
 *
 * Quand la camera regarde vers -Z avec un vecteur up en Y, on s'attend a ce que
 * w pointe vers +Z, u vers +X et v vers +Y.
 */
static bool test_camera_coordinate_system() {
    Camera cam;
    cam.initialize(800, 600, Point3(0, 0, 0), Point3(0, 0, -1), Vec3(0, 1, 0), 90.0f);
    TEST_ASSERT_NEAR(cam.w.z, 1.0f, 1e-5f);
    TEST_ASSERT_NEAR(cam.u.x, 1.0f, 1e-5f);
    TEST_ASSERT_NEAR(cam.v.y, 1.0f, 1e-5f);
    return true;
}

/**
 * @brief Verifie que l'angle de defocus est nul quand aucun flou n'est demande.
 */
static bool test_camera_no_defocus() {
    Camera cam;
    cam.initialize(800, 600, Point3(0, 0, 0), Point3(0, 0, -1), Vec3(0, 1, 0), 90.0f, 0.0f);
    TEST_ASSERT_NEAR(cam.defocus_angle, 0.0f, 1e-5f);
    return true;
}

/**
 * @brief Verifie que le defocus blur est actif quand un angle de defocus non nul est fourni.
 *
 * On verifie que les vecteurs du disque de defocus ont une longueur non nulle,
 * ce qui signifie que les rayons seront perturbes pour simuler la profondeur de champ.
 */
static bool test_camera_with_defocus() {
    Camera cam;
    cam.initialize(800, 600, Point3(0, 0, 0), Point3(0, 0, -1), Vec3(0, 1, 0), 90.0f, 2.0f, 10.0f);
    TEST_ASSERT_NEAR(cam.defocus_angle, 2.0f, 1e-5f);
    TEST_ASSERT(cam.defocus_disk_u.length() > 0.0f);
    TEST_ASSERT(cam.defocus_disk_v.length() > 0.0f);
    return true;
}


/**
 * @brief Verifie qu'un rayon genere depuis le pixel central part bien de l'origine de la camera.
 *
 * Le rayon doit avoir son origine au centre de la camera et sa direction
 * doit pointer globalement vers -Z (vers la scene).
 */
static bool test_camera_ray_from_center() {
    Camera cam;
    cam.initialize(800, 600, Point3(0, 0, 0), Point3(0, 0, -1), Vec3(0, 1, 0), 90.0f);
    CPURandom rng(42);
    Ray r = cam.get_ray_cpu(400, 300, rng);
    TEST_ASSERT_NEAR(r.origin().x, 0.0f, 1e-5f);
    TEST_ASSERT_NEAR(r.origin().y, 0.0f, 1e-5f);
    TEST_ASSERT_NEAR(r.origin().z, 0.0f, 1e-5f);
    Vec3 dir = unit_vector(r.direction());
    TEST_ASSERT(dir.z < -0.5f);
    return true;
}

/**
 * @brief Verifie que les rayons des coins opposes de l'image divergent suffisamment.
 *
 * Deux rayons tires depuis les coins (0,0) et (799,599) doivent avoir des
 * directions significativement differentes (cosinus < 0.9), ce qui confirme
 * que le champ de vision est bien pris en compte.
 */
static bool test_camera_ray_corner_diverges() {
    Camera cam;
    cam.initialize(800, 600, Point3(0, 0, 0), Point3(0, 0, -1), Vec3(0, 1, 0), 90.0f);
    CPURandom rng(42);
    Ray r1 = cam.get_ray_cpu(0, 0, rng);
    Ray r2 = cam.get_ray_cpu(799, 599, rng);
    Vec3 d1 = unit_vector(r1.direction());
    Vec3 d2 = unit_vector(r2.direction());
    float cosine = dot(d1, d2);
    TEST_ASSERT(cosine < 0.9f);
    return true;
}

/**
 * @brief Verifie que le temps du rayon est compris dans l'intervalle d'obturation [0, 1].
 *
 * Ce test s'assure que le motion blur fonctionne correctement en verifiant
 * que le temps attribue au rayon est bien borne.
 */
static bool test_camera_shutter_time() {
    Camera cam;
    cam.initialize(800, 600, Point3(0, 0, 0), Point3(0, 0, -1), Vec3(0, 1, 0), 90.0f,
                   0.0f, 10.0f, 0.0f, 1.0f);
    CPURandom rng(42);
    Ray r = cam.get_ray_cpu(400, 300, rng);
    TEST_ASSERT(r.time() >= 0.0f);
    TEST_ASSERT(r.time() <= 1.0f);
    return true;
}

/**
 * @brief Verifie que le champ de vision (FOV) affecte bien l'ecartement des rayons.
 *
 * Un FOV large (90 degres) doit produire un delta pixel plus grand qu'un FOV
 * etroit (20 degres), ce qui confirme la bonne prise en compte du FOV.
 */
static bool test_camera_fov_affects_spread() {
    Camera cam_narrow, cam_wide;
    cam_narrow.initialize(800, 600, Point3(0, 0, 0), Point3(0, 0, -1), Vec3(0, 1, 0), 20.0f);
    cam_wide.initialize(800, 600, Point3(0, 0, 0), Point3(0, 0, -1), Vec3(0, 1, 0), 90.0f);
    float narrow_delta = cam_narrow.pixel_delta_u.length();
    float wide_delta = cam_wide.pixel_delta_u.length();
    TEST_ASSERT(wide_delta > narrow_delta);
    return true;
}


/**
 * @brief Lance l'ensemble des tests unitaires de la camera.
 *
 * @param passed Compteur de tests reussis (mis a jour par reference).
 * @param failed Compteur de tests echoues (mis a jour par reference).
 * @param total  Compteur total de tests executes (mis a jour par reference).
 */
void run_camera_tests(int& passed, int& failed, int& total) {
    RUN_TEST(test_camera_center);
    RUN_TEST(test_camera_dimensions);
    RUN_TEST(test_camera_coordinate_system);
    RUN_TEST(test_camera_no_defocus);
    RUN_TEST(test_camera_with_defocus);
    RUN_TEST(test_camera_ray_from_center);
    RUN_TEST(test_camera_ray_corner_diverges);
    RUN_TEST(test_camera_shutter_time);
    RUN_TEST(test_camera_fov_affects_spread);
}
