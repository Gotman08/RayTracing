/**
 * @file test_tone_mapping.cpp
 * @brief Tests unitaires pour le tone mapping Reinhard et la correction gamma.
 *
 * Ce fichier verifie le bon fonctionnement de la conversion HDR vers LDR
 * via l'operateur de tone mapping Reinhard, ainsi que la correction gamma
 * appliquee avant l'affichage. On teste les cas limites (noir, blanc),
 * les valeurs elevees, la monotonie, et le clamping des valeurs negatives.
 */

#include <iostream>
#include <cmath>

#include "raytracer/core/vec3.cuh"
#include "raytracer/rendering/tone_mapping.cuh"

using namespace rt;

/**
 * @brief Macro d'assertion simple pour les tests.
 *
 * Verifie qu'une expression est vraie, sinon affiche un message d'erreur.
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
 * @brief Verifie que le tone mapping d'une couleur noire (0,0,0) reste noire.
 *
 * L'operateur Reinhard applique c/(1+c), donc 0/(1+0) = 0.
 */
static bool test_tone_mapping_zero() {
    Color result = apply_tone_mapping(Color(0, 0, 0));
    TEST_ASSERT_NEAR(result.x, 0.0f, 1e-6f);
    TEST_ASSERT_NEAR(result.y, 0.0f, 1e-6f);
    TEST_ASSERT_NEAR(result.z, 0.0f, 1e-6f);
    return true;
}

/**
 * @brief Verifie que le tone mapping de (1,1,1) donne (0.5, 0.5, 0.5).
 *
 * Avec Reinhard : 1/(1+1) = 0.5. C'est un cas de reference simple.
 */
static bool test_tone_mapping_one() {
    Color result = apply_tone_mapping(Color(1, 1, 1));
    TEST_ASSERT_NEAR(result.x, 0.5f, 1e-6f);
    TEST_ASSERT_NEAR(result.y, 0.5f, 1e-6f);
    TEST_ASSERT_NEAR(result.z, 0.5f, 1e-6f);
    return true;
}

/**
 * @brief Verifie que le tone mapping de valeurs HDR tres elevees reste en dessous de 1.
 *
 * Pour une valeur de 100, Reinhard donne 100/101 ~ 0.99, ce qui montre
 * que la fonction compresse bien les hautes lumieres sans depasser 1.
 */
static bool test_tone_mapping_high_value() {
    Color result = apply_tone_mapping(Color(100, 100, 100));
    TEST_ASSERT(result.x < 1.0f);
    TEST_ASSERT(result.x > 0.98f);
    return true;
}

/**
 * @brief Verifie les valeurs exactes du tone mapping Reinhard par composante.
 *
 * Pour (2, 4, 8), on attend respectivement 2/3, 4/5, 8/9 selon la formule c/(1+c).
 */
static bool test_tone_mapping_preserves_ratios() {
    Color hdr(2.0f, 4.0f, 8.0f);
    Color result = apply_tone_mapping(hdr);
    TEST_ASSERT_NEAR(result.x, 2.0f / 3.0f, 1e-5f);
    TEST_ASSERT_NEAR(result.y, 4.0f / 5.0f, 1e-5f);
    TEST_ASSERT_NEAR(result.z, 8.0f / 9.0f, 1e-5f);
    return true;
}

/**
 * @brief Verifie que le tone mapping est une fonction monotone croissante.
 *
 * Une couleur plus lumineuse en entree doit donner une couleur plus lumineuse
 * en sortie. C'est essentiel pour conserver l'ordre des intensites.
 */
static bool test_tone_mapping_monotonic() {
    Color dim = apply_tone_mapping(Color(0.5f, 0.5f, 0.5f));
    Color bright = apply_tone_mapping(Color(2.0f, 2.0f, 2.0f));
    TEST_ASSERT(bright.x > dim.x);
    return true;
}


/**
 * @brief Verifie que la correction gamma d'une couleur noire reste noire.
 *
 * 0^(1/gamma) = 0 quel que soit le gamma.
 */
static bool test_gamma_correction_zero() {
    Color result = gamma_correct(Color(0, 0, 0));
    TEST_ASSERT_NEAR(result.x, 0.0f, 1e-6f);
    TEST_ASSERT_NEAR(result.y, 0.0f, 1e-6f);
    TEST_ASSERT_NEAR(result.z, 0.0f, 1e-6f);
    return true;
}

/**
 * @brief Verifie que la correction gamma de (1,1,1) reste (1,1,1).
 *
 * 1^(1/gamma) = 1 quel que soit le gamma.
 */
static bool test_gamma_correction_one() {
    Color result = gamma_correct(Color(1, 1, 1));
    TEST_ASSERT_NEAR(result.x, 1.0f, 1e-5f);
    TEST_ASSERT_NEAR(result.y, 1.0f, 1e-5f);
    TEST_ASSERT_NEAR(result.z, 1.0f, 1e-5f);
    return true;
}

/**
 * @brief Verifie la correction gamma pour un ton moyen (0.5) avec gamma 2.2.
 *
 * Le resultat attendu est 0.5^(1/2.2) ~ 0.7297. Cela correspond a
 * l'eclaircissement typique du gamma sRGB.
 */
static bool test_gamma_correction_midtone() {
    Color result = gamma_correct(Color(0.5f, 0.5f, 0.5f), 2.2f);
    float expected = powf(0.5f, 1.0f / 2.2f);
    TEST_ASSERT_NEAR(result.x, expected, 1e-4f);
    return true;
}

/**
 * @brief Verifie que la correction gamma eclaircit les tons sombres.
 *
 * Pour une valeur lineaire de 0.25, la valeur corrigee doit etre superieure
 * car le gamma < 1 (1/2.2) etire les valeurs sombres vers le haut.
 */
static bool test_gamma_correction_brightens_darks() {
    Color linear(0.25f, 0.25f, 0.25f);
    Color corrected = gamma_correct(linear);
    TEST_ASSERT(corrected.x > linear.x);
    return true;
}

/**
 * @brief Verifie que les valeurs negatives sont clampees a zero avant la correction gamma.
 *
 * Des valeurs negatives n'ont pas de sens physique pour la lumiere,
 * elles doivent etre ramenees a 0.
 */
static bool test_gamma_correction_negative_clamped() {
    Color result = gamma_correct(Color(-1.0f, -0.5f, 0.0f));
    TEST_ASSERT_NEAR(result.x, 0.0f, 1e-6f);
    TEST_ASSERT_NEAR(result.y, 0.0f, 1e-6f);
    TEST_ASSERT_NEAR(result.z, 0.0f, 1e-6f);
    return true;
}

/**
 * @brief Verifie la correction gamma avec gamma = 2.0 pour une valeur connue.
 *
 * Avec gamma 2.0, la correction est simplement la racine carree :
 * 0.25^(1/2) = 0.5. C'est un cas de validation analytique simple.
 */
static bool test_gamma_2_known_value() {
    Color result = gamma_correct(Color(0.25f, 0.25f, 0.25f), 2.0f);
    TEST_ASSERT_NEAR(result.x, 0.5f, 1e-5f);
    return true;
}


/**
 * @brief Lance l'ensemble des tests unitaires du tone mapping et de la correction gamma.
 *
 * @param passed Compteur de tests reussis (mis a jour par reference).
 * @param failed Compteur de tests echoues (mis a jour par reference).
 * @param total  Compteur total de tests executes (mis a jour par reference).
 */
void run_tone_mapping_tests(int& passed, int& failed, int& total) {
    RUN_TEST(test_tone_mapping_zero);
    RUN_TEST(test_tone_mapping_one);
    RUN_TEST(test_tone_mapping_high_value);
    RUN_TEST(test_tone_mapping_preserves_ratios);
    RUN_TEST(test_tone_mapping_monotonic);
    RUN_TEST(test_gamma_correction_zero);
    RUN_TEST(test_gamma_correction_one);
    RUN_TEST(test_gamma_correction_midtone);
    RUN_TEST(test_gamma_correction_brightens_darks);
    RUN_TEST(test_gamma_correction_negative_clamped);
    RUN_TEST(test_gamma_2_known_value);
}
