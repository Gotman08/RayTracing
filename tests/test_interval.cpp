/**
 * @file test_interval.cpp
 * @brief Tests unitaires pour la classe Interval
 *
 * Ce fichier contient les tests de la classe Interval qui represente
 * un intervalle ferme [min, max]. Cette structure est tres utilisee
 * en raytracing pour definir les bornes valides du parametre t d'un rayon,
 * ainsi que pour les boites englobantes (AABB). On teste les constructeurs,
 * le calcul de taille, les fonctions contains/surrounds, le clamp
 * et l'expansion/retrecissement de l'intervalle.
 */

#include <iostream>
#include <cmath>
#include <cfloat>

#include "raytracer/core/interval.cuh"

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

constexpr float EPS = 1e-6f;


/**
 * @brief Teste le constructeur par defaut d'Interval
 *
 * Le constructeur par defaut doit creer un intervalle vide,
 * c'est-a-dire ou min > max (convention pour un intervalle invalide).
 */
bool test_interval_default_constructor() {
    Interval i;
    TEST_ASSERT(i.min > i.max);
    return true;
}

/**
 * @brief Teste le constructeur avec valeurs min et max explicites
 *
 * Verifie que Interval(1, 5) stocke correctement les bornes.
 */
bool test_interval_value_constructor() {
    Interval i(1.0f, 5.0f);
    TEST_ASSERT_NEAR(i.min, 1.0f, EPS);
    TEST_ASSERT_NEAR(i.max, 5.0f, EPS);
    return true;
}

/**
 * @brief Teste le constructeur de fusion de deux intervalles qui se chevauchent
 *
 * La fusion de [1,3] et [2,5] doit donner [1,5], englobant les deux intervalles.
 */
bool test_interval_merge_constructor() {
    Interval a(1.0f, 3.0f);
    Interval b(2.0f, 5.0f);
    Interval merged(a, b);
    TEST_ASSERT_NEAR(merged.min, 1.0f, EPS);
    TEST_ASSERT_NEAR(merged.max, 5.0f, EPS);
    return true;
}

/**
 * @brief Teste la fusion de deux intervalles disjoints
 *
 * La fusion de [1,2] et [4,5] doit donner [1,5], couvrant
 * le plus petit min et le plus grand max meme si les intervalles
 * ne se chevauchent pas.
 */
bool test_interval_merge_disjoint() {
    Interval a(1.0f, 2.0f);
    Interval b(4.0f, 5.0f);
    Interval merged(a, b);
    TEST_ASSERT_NEAR(merged.min, 1.0f, EPS);
    TEST_ASSERT_NEAR(merged.max, 5.0f, EPS);
    return true;
}


/**
 * @brief Teste le calcul de la taille de l'intervalle
 *
 * La taille de [2, 7] doit etre 5 (max - min).
 */
bool test_interval_size() {
    Interval i(2.0f, 7.0f);
    TEST_ASSERT_NEAR(i.size(), 5.0f, EPS);
    return true;
}

/**
 * @brief Teste la taille d'un intervalle degenere (min == max)
 *
 * Un intervalle [3, 3] a une taille de 0.
 */
bool test_interval_size_zero() {
    Interval i(3.0f, 3.0f);
    TEST_ASSERT_NEAR(i.size(), 0.0f, EPS);
    return true;
}

/**
 * @brief Teste la taille d'un intervalle inverse (min > max)
 *
 * Un intervalle [5, 2] a une taille negative, ce qui indique
 * un intervalle invalide ou vide.
 */
bool test_interval_size_negative() {
    Interval i(5.0f, 2.0f);
    TEST_ASSERT(i.size() < 0.0f);
    return true;
}


/**
 * @brief Teste contains() avec une valeur a l'interieur de l'intervalle
 *
 * La valeur 5 doit etre contenue dans [0, 10].
 */
bool test_interval_contains_inside() {
    Interval i(0.0f, 10.0f);
    TEST_ASSERT(i.contains(5.0f) == true);
    return true;
}

/**
 * @brief Teste contains() sur la borne inferieure
 *
 * La valeur min (0) doit etre consideree comme contenue (intervalle ferme).
 */
bool test_interval_contains_at_min() {
    Interval i(0.0f, 10.0f);
    TEST_ASSERT(i.contains(0.0f) == true);
    return true;
}

/**
 * @brief Teste contains() sur la borne superieure
 *
 * La valeur max (10) doit etre consideree comme contenue (intervalle ferme).
 */
bool test_interval_contains_at_max() {
    Interval i(0.0f, 10.0f);
    TEST_ASSERT(i.contains(10.0f) == true);
    return true;
}

/**
 * @brief Teste contains() avec une valeur en dessous de l'intervalle
 *
 * La valeur -1 ne doit pas etre contenue dans [0, 10].
 */
bool test_interval_contains_outside_below() {
    Interval i(0.0f, 10.0f);
    TEST_ASSERT(i.contains(-1.0f) == false);
    return true;
}

/**
 * @brief Teste contains() avec une valeur au dessus de l'intervalle
 *
 * La valeur 11 ne doit pas etre contenue dans [0, 10].
 */
bool test_interval_contains_outside_above() {
    Interval i(0.0f, 10.0f);
    TEST_ASSERT(i.contains(11.0f) == false);
    return true;
}


/**
 * @brief Teste surrounds() avec une valeur strictement a l'interieur
 *
 * Contrairement a contains(), surrounds() exclut les bornes (intervalle ouvert).
 * La valeur 5 est strictement dans ]0, 10[.
 */
bool test_interval_surrounds_inside() {
    Interval i(0.0f, 10.0f);
    TEST_ASSERT(i.surrounds(5.0f) == true);
    return true;
}

/**
 * @brief Teste surrounds() sur la borne inferieure (doit echouer)
 *
 * La valeur min (0) n'est pas strictement a l'interieur de ]0, 10[.
 */
bool test_interval_surrounds_at_min() {
    Interval i(0.0f, 10.0f);
    TEST_ASSERT(i.surrounds(0.0f) == false);
    return true;
}

/**
 * @brief Teste surrounds() sur la borne superieure (doit echouer)
 *
 * La valeur max (10) n'est pas strictement a l'interieur de ]0, 10[.
 */
bool test_interval_surrounds_at_max() {
    Interval i(0.0f, 10.0f);
    TEST_ASSERT(i.surrounds(10.0f) == false);
    return true;
}

/**
 * @brief Teste surrounds() avec une valeur juste au dessus de min
 *
 * La valeur 0.001 est strictement superieure a 0, donc elle doit
 * etre entouree par l'intervalle ]0, 10[.
 */
bool test_interval_surrounds_near_min() {
    Interval i(0.0f, 10.0f);
    TEST_ASSERT(i.surrounds(0.001f) == true);
    return true;
}


/**
 * @brief Teste clamp() avec une valeur deja dans l'intervalle
 *
 * Une valeur interieure ne doit pas etre modifiee par le clamp.
 */
bool test_interval_clamp_inside() {
    Interval i(0.0f, 10.0f);
    TEST_ASSERT_NEAR(i.clamp(5.0f), 5.0f, EPS);
    return true;
}

/**
 * @brief Teste clamp() avec une valeur en dessous de l'intervalle
 *
 * Une valeur inferieure a min doit etre ramenee a min.
 */
bool test_interval_clamp_below() {
    Interval i(0.0f, 10.0f);
    TEST_ASSERT_NEAR(i.clamp(-5.0f), 0.0f, EPS);
    return true;
}

/**
 * @brief Teste clamp() avec une valeur au dessus de l'intervalle
 *
 * Une valeur superieure a max doit etre ramenee a max.
 */
bool test_interval_clamp_above() {
    Interval i(0.0f, 10.0f);
    TEST_ASSERT_NEAR(i.clamp(15.0f), 10.0f, EPS);
    return true;
}

/**
 * @brief Teste clamp() sur les bornes exactes de l'intervalle
 *
 * Les valeurs min et max doivent rester inchangees apres le clamp.
 */
bool test_interval_clamp_at_boundary() {
    Interval i(0.0f, 10.0f);
    TEST_ASSERT_NEAR(i.clamp(0.0f), 0.0f, EPS);
    TEST_ASSERT_NEAR(i.clamp(10.0f), 10.0f, EPS);
    return true;
}


/**
 * @brief Teste l'expansion d'un intervalle
 *
 * L'expansion de [2, 8] par 2 doit donner [1, 9] car on ajoute
 * delta/2 de chaque cote.
 */
bool test_interval_expand() {
    Interval i(2.0f, 8.0f);
    Interval expanded = i.expand(2.0f);
    TEST_ASSERT_NEAR(expanded.min, 1.0f, EPS);
    TEST_ASSERT_NEAR(expanded.max, 9.0f, EPS);
    return true;
}

/**
 * @brief Teste l'expansion par zero (aucun changement attendu)
 *
 * L'intervalle doit rester identique si delta vaut 0.
 */
bool test_interval_expand_zero() {
    Interval i(2.0f, 8.0f);
    Interval expanded = i.expand(0.0f);
    TEST_ASSERT_NEAR(expanded.min, 2.0f, EPS);
    TEST_ASSERT_NEAR(expanded.max, 8.0f, EPS);
    return true;
}

/**
 * @brief Teste l'expansion avec une valeur negative (retrecissement)
 *
 * Un delta negatif retrecit l'intervalle : [0, 10] avec delta=-4
 * doit donner [2, 8].
 */
bool test_interval_expand_negative() {
    Interval i(0.0f, 10.0f);
    Interval shrunk = i.expand(-4.0f);
    TEST_ASSERT_NEAR(shrunk.min, 2.0f, EPS);
    TEST_ASSERT_NEAR(shrunk.max, 8.0f, EPS);
    return true;
}


/**
 * @brief Execute l'ensemble des tests unitaires pour Interval
 *
 * Lance tous les tests de la classe Interval et met a jour les compteurs.
 *
 * @param passed Compteur de tests reussis
 * @param failed Compteur de tests echoues
 * @param total Compteur du nombre total de tests executes
 */
void run_interval_tests(int& passed, int& failed, int& total) {
    RUN_TEST(test_interval_default_constructor);
    RUN_TEST(test_interval_value_constructor);
    RUN_TEST(test_interval_merge_constructor);
    RUN_TEST(test_interval_merge_disjoint);

    RUN_TEST(test_interval_size);
    RUN_TEST(test_interval_size_zero);
    RUN_TEST(test_interval_size_negative);

    RUN_TEST(test_interval_contains_inside);
    RUN_TEST(test_interval_contains_at_min);
    RUN_TEST(test_interval_contains_at_max);
    RUN_TEST(test_interval_contains_outside_below);
    RUN_TEST(test_interval_contains_outside_above);

    RUN_TEST(test_interval_surrounds_inside);
    RUN_TEST(test_interval_surrounds_at_min);
    RUN_TEST(test_interval_surrounds_at_max);
    RUN_TEST(test_interval_surrounds_near_min);

    RUN_TEST(test_interval_clamp_inside);
    RUN_TEST(test_interval_clamp_below);
    RUN_TEST(test_interval_clamp_above);
    RUN_TEST(test_interval_clamp_at_boundary);

    RUN_TEST(test_interval_expand);
    RUN_TEST(test_interval_expand_zero);
    RUN_TEST(test_interval_expand_negative);
}
