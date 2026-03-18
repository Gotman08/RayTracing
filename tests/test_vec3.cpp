/**
 * @file test_vec3.cpp
 * @brief Tests unitaires pour la classe Vec3 (vecteur 3D)
 *
 * Ce fichier contient l'ensemble des tests unitaires permettant de valider
 * le bon fonctionnement de la classe Vec3. On y verifie les constructeurs,
 * les operateurs arithmetiques (+, -, *, /), les operateurs composes (+=, *=, /=),
 * ainsi que les fonctions mathematiques essentielles au raytracing :
 * produit scalaire (dot), produit vectoriel (cross), reflexion, refraction,
 * normalisation et detection de vecteurs quasi-nuls.
 * Les alias Point3 et Color sont egalement testes.
 */

#include <iostream>
#include <cmath>

#include "raytracer/core/vec3.cuh"

using namespace rt;

extern int passed, failed, total;

/**
 * @brief Macro d'assertion simple pour les tests
 *
 * Verifie qu'une expression booleenne est vraie. En cas d'echec,
 * affiche l'expression fautive ainsi que le fichier et la ligne correspondante,
 * puis retourne false pour signaler l'echec du test.
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
 * Compare deux valeurs flottantes en verifiant que leur difference absolue
 * ne depasse pas la tolerance eps. Indispensable pour eviter les faux negatifs
 * dus aux erreurs d'arrondi en virgule flottante.
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
 * Execute la fonction de test passee en parametre, affiche le resultat
 * ([PASS] ou [FAIL]) et met a jour les compteurs de tests reussis,
 * echoues et totaux.
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
 * @brief Teste le constructeur par defaut de Vec3
 *
 * Verifie que le constructeur sans argument initialise bien
 * toutes les composantes (x, y, z) a zero.
 */
bool test_vec3_default_constructor() {
    Vec3 v;
    TEST_ASSERT_NEAR(v.x, 0.0f, EPS);
    TEST_ASSERT_NEAR(v.y, 0.0f, EPS);
    TEST_ASSERT_NEAR(v.z, 0.0f, EPS);
    return true;
}

/**
 * @brief Teste le constructeur avec trois valeurs distinctes
 *
 * Verifie que Vec3(1, 2, 3) stocke bien chaque composante
 * dans le membre correspondant.
 */
bool test_vec3_value_constructor() {
    Vec3 v(1.0f, 2.0f, 3.0f);
    TEST_ASSERT_NEAR(v.x, 1.0f, EPS);
    TEST_ASSERT_NEAR(v.y, 2.0f, EPS);
    TEST_ASSERT_NEAR(v.z, 3.0f, EPS);
    return true;
}

/**
 * @brief Teste le constructeur avec une seule valeur
 *
 * Verifie que Vec3(5) initialise les trois composantes a 5,
 * ce qui est utile pour creer des vecteurs uniformes.
 */
bool test_vec3_single_value_constructor() {
    Vec3 v(5.0f);
    TEST_ASSERT_NEAR(v.x, 5.0f, EPS);
    TEST_ASSERT_NEAR(v.y, 5.0f, EPS);
    TEST_ASSERT_NEAR(v.z, 5.0f, EPS);
    return true;
}


/**
 * @brief Teste l'operateur de negation unaire (-Vec3)
 *
 * Verifie que la negation inverse le signe de chaque composante.
 */
bool test_vec3_negation() {
    Vec3 v(1.0f, -2.0f, 3.0f);
    Vec3 neg = -v;
    TEST_ASSERT_NEAR(neg.x, -1.0f, EPS);
    TEST_ASSERT_NEAR(neg.y, 2.0f, EPS);
    TEST_ASSERT_NEAR(neg.z, -3.0f, EPS);
    return true;
}

/**
 * @brief Teste l'addition de deux vecteurs (operator+)
 *
 * Verifie que l'addition se fait composante par composante.
 */
bool test_vec3_addition() {
    Vec3 a(1.0f, 2.0f, 3.0f);
    Vec3 b(4.0f, 5.0f, 6.0f);
    Vec3 c = a + b;
    TEST_ASSERT_NEAR(c.x, 5.0f, EPS);
    TEST_ASSERT_NEAR(c.y, 7.0f, EPS);
    TEST_ASSERT_NEAR(c.z, 9.0f, EPS);
    return true;
}

/**
 * @brief Teste la soustraction de deux vecteurs (operator-)
 *
 * Verifie que la soustraction se fait composante par composante.
 */
bool test_vec3_subtraction() {
    Vec3 a(4.0f, 5.0f, 6.0f);
    Vec3 b(1.0f, 2.0f, 3.0f);
    Vec3 c = a - b;
    TEST_ASSERT_NEAR(c.x, 3.0f, EPS);
    TEST_ASSERT_NEAR(c.y, 3.0f, EPS);
    TEST_ASSERT_NEAR(c.z, 3.0f, EPS);
    return true;
}

/**
 * @brief Teste la multiplication composante par composante de deux vecteurs
 *
 * Verifie que chaque composante du resultat est le produit des composantes correspondantes.
 */
bool test_vec3_multiplication() {
    Vec3 a(1.0f, 2.0f, 3.0f);
    Vec3 b(2.0f, 3.0f, 4.0f);
    Vec3 c = a * b;
    TEST_ASSERT_NEAR(c.x, 2.0f, EPS);
    TEST_ASSERT_NEAR(c.y, 6.0f, EPS);
    TEST_ASSERT_NEAR(c.z, 12.0f, EPS);
    return true;
}

/**
 * @brief Teste la multiplication par un scalaire (Vec3 * float et float * Vec3)
 *
 * Verifie la commutativite de la multiplication scalaire :
 * v * 2 doit donner le meme resultat que 2 * v.
 */
bool test_vec3_scalar_multiplication() {
    Vec3 v(1.0f, 2.0f, 3.0f);
    Vec3 c1 = v * 2.0f;
    Vec3 c2 = 2.0f * v;
    TEST_ASSERT_NEAR(c1.x, 2.0f, EPS);
    TEST_ASSERT_NEAR(c1.y, 4.0f, EPS);
    TEST_ASSERT_NEAR(c1.z, 6.0f, EPS);
    TEST_ASSERT_NEAR(c2.x, 2.0f, EPS);
    TEST_ASSERT_NEAR(c2.y, 4.0f, EPS);
    TEST_ASSERT_NEAR(c2.z, 6.0f, EPS);
    return true;
}

/**
 * @brief Teste la division d'un vecteur par un scalaire
 *
 * Verifie que chaque composante est bien divisee par le scalaire.
 */
bool test_vec3_division() {
    Vec3 v(2.0f, 4.0f, 6.0f);
    Vec3 c = v / 2.0f;
    TEST_ASSERT_NEAR(c.x, 1.0f, EPS);
    TEST_ASSERT_NEAR(c.y, 2.0f, EPS);
    TEST_ASSERT_NEAR(c.z, 3.0f, EPS);
    return true;
}

/**
 * @brief Teste l'operateur d'indexation (operator[])
 *
 * Verifie que v[0], v[1], v[2] retournent respectivement x, y et z.
 */
bool test_vec3_index_operator() {
    Vec3 v(1.0f, 2.0f, 3.0f);
    TEST_ASSERT_NEAR(v[0], 1.0f, EPS);
    TEST_ASSERT_NEAR(v[1], 2.0f, EPS);
    TEST_ASSERT_NEAR(v[2], 3.0f, EPS);
    return true;
}

/**
 * @brief Teste l'operateur d'addition composee (operator+=)
 *
 * Verifie que l'ajout en place modifie correctement le vecteur.
 */
bool test_vec3_compound_addition() {
    Vec3 v(1.0f, 2.0f, 3.0f);
    v += Vec3(1.0f, 1.0f, 1.0f);
    TEST_ASSERT_NEAR(v.x, 2.0f, EPS);
    TEST_ASSERT_NEAR(v.y, 3.0f, EPS);
    TEST_ASSERT_NEAR(v.z, 4.0f, EPS);
    return true;
}

/**
 * @brief Teste l'operateur de multiplication composee (operator*=)
 *
 * Verifie que la multiplication en place par un scalaire fonctionne.
 */
bool test_vec3_compound_multiplication() {
    Vec3 v(1.0f, 2.0f, 3.0f);
    v *= 2.0f;
    TEST_ASSERT_NEAR(v.x, 2.0f, EPS);
    TEST_ASSERT_NEAR(v.y, 4.0f, EPS);
    TEST_ASSERT_NEAR(v.z, 6.0f, EPS);
    return true;
}

/**
 * @brief Teste l'operateur de division composee (operator/=)
 *
 * Verifie que la division en place par un scalaire fonctionne.
 */
bool test_vec3_compound_division() {
    Vec3 v(2.0f, 4.0f, 6.0f);
    v /= 2.0f;
    TEST_ASSERT_NEAR(v.x, 1.0f, EPS);
    TEST_ASSERT_NEAR(v.y, 2.0f, EPS);
    TEST_ASSERT_NEAR(v.z, 3.0f, EPS);
    return true;
}


/**
 * @brief Teste le calcul de la norme (longueur) du vecteur
 *
 * Utilise le triangle classique (3, 4, 0) dont la norme vaut 5.
 */
bool test_vec3_length() {
    Vec3 v(3.0f, 4.0f, 0.0f);
    TEST_ASSERT_NEAR(v.length(), 5.0f, EPS);
    return true;
}

/**
 * @brief Teste le calcul de la norme au carre
 *
 * La norme au carre evite la racine carree et vaut ici 25.
 * Fonction souvent utilisee pour des comparaisons de distance sans sqrt.
 */
bool test_vec3_length_squared() {
    Vec3 v(3.0f, 4.0f, 0.0f);
    TEST_ASSERT_NEAR(v.length_squared(), 25.0f, EPS);
    return true;
}

/**
 * @brief Teste la normalisation du vecteur (vecteur unitaire)
 *
 * Verifie que le vecteur normalise a une longueur de 1
 * et que ses composantes sont correctes (0.6, 0.8, 0.0).
 */
bool test_vec3_normalized() {
    Vec3 v(3.0f, 4.0f, 0.0f);
    Vec3 n = v.normalized();
    TEST_ASSERT_NEAR(n.length(), 1.0f, EPS);
    TEST_ASSERT_NEAR(n.x, 0.6f, EPS);
    TEST_ASSERT_NEAR(n.y, 0.8f, EPS);
    TEST_ASSERT_NEAR(n.z, 0.0f, EPS);
    return true;
}

/**
 * @brief Teste la detection de vecteurs quasi-nuls (near_zero)
 *
 * Un vecteur tres petit (composantes ~1e-9) doit etre detecte comme quasi-nul,
 * tandis qu'un vecteur standard ne l'est pas. Utile pour eviter
 * les divisions par zero dans les calculs de reflexion/refraction.
 */
bool test_vec3_near_zero() {
    Vec3 small(1e-9f, 1e-9f, 1e-9f);
    Vec3 large(1.0f, 0.0f, 0.0f);
    TEST_ASSERT(small.near_zero() == true);
    TEST_ASSERT(large.near_zero() == false);
    return true;
}


/**
 * @brief Teste le produit scalaire (dot product)
 *
 * Verifie que dot((1,2,3), (4,5,6)) = 1*4 + 2*5 + 3*6 = 32.
 * Le produit scalaire est fondamental pour les calculs d'eclairage.
 */
bool test_vec3_dot() {
    Vec3 a(1.0f, 2.0f, 3.0f);
    Vec3 b(4.0f, 5.0f, 6.0f);
    float d = dot(a, b);
    TEST_ASSERT_NEAR(d, 32.0f, EPS);
    return true;
}

/**
 * @brief Teste le produit vectoriel (cross product)
 *
 * Verifie que cross(x, y) = z selon la regle de la main droite.
 * Le produit vectoriel donne un vecteur perpendiculaire aux deux operandes.
 */
bool test_vec3_cross() {
    Vec3 a(1.0f, 0.0f, 0.0f);
    Vec3 b(0.0f, 1.0f, 0.0f);
    Vec3 c = cross(a, b);
    TEST_ASSERT_NEAR(c.x, 0.0f, EPS);
    TEST_ASSERT_NEAR(c.y, 0.0f, EPS);
    TEST_ASSERT_NEAR(c.z, 1.0f, EPS);
    return true;
}

/**
 * @brief Teste l'anticommutativite du produit vectoriel
 *
 * Verifie que cross(a, b) = -cross(b, a), propriete fondamentale
 * du produit vectoriel.
 */
bool test_vec3_cross_anticommutative() {
    Vec3 a(1.0f, 0.0f, 0.0f);
    Vec3 b(0.0f, 1.0f, 0.0f);
    Vec3 c1 = cross(a, b);
    Vec3 c2 = cross(b, a);
    TEST_ASSERT_NEAR(c1.x, -c2.x, EPS);
    TEST_ASSERT_NEAR(c1.y, -c2.y, EPS);
    TEST_ASSERT_NEAR(c1.z, -c2.z, EPS);
    return true;
}

/**
 * @brief Teste la reflexion d'un vecteur par rapport a une normale
 *
 * Un vecteur incident (1, -1, 0) reflechi par la normale (0, 1, 0)
 * doit donner (1, 1, 0). Essentiel pour les materiaux metalliques.
 */
bool test_vec3_reflect() {
    Vec3 v(1.0f, -1.0f, 0.0f);
    Vec3 n(0.0f, 1.0f, 0.0f);
    Vec3 r = reflect(v, n);
    TEST_ASSERT_NEAR(r.x, 1.0f, EPS);
    TEST_ASSERT_NEAR(r.y, 1.0f, EPS);
    TEST_ASSERT_NEAR(r.z, 0.0f, EPS);
    return true;
}

/**
 * @brief Teste la refraction d'un rayon perpendiculaire a la surface
 *
 * Avec un indice de refraction de 1.0 et une incidence perpendiculaire,
 * le rayon refracte doit garder la meme direction. Cela correspond
 * au cas trivial de la loi de Snell-Descartes.
 */
bool test_vec3_refract_perpendicular() {
    Vec3 v(0.0f, -1.0f, 0.0f);
    Vec3 n(0.0f, 1.0f, 0.0f);
    Vec3 r = refract(v, n, 1.0f);
    TEST_ASSERT_NEAR(r.x, 0.0f, EPS);
    TEST_ASSERT_NEAR(r.y, -1.0f, EPS);
    TEST_ASSERT_NEAR(r.z, 0.0f, EPS);
    return true;
}

/**
 * @brief Teste la fonction libre unit_vector
 *
 * Verifie que unit_vector retourne un vecteur de norme 1.
 */
bool test_vec3_unit_vector() {
    Vec3 v(3.0f, 4.0f, 0.0f);
    Vec3 u = unit_vector(v);
    TEST_ASSERT_NEAR(u.length(), 1.0f, EPS);
    return true;
}


/**
 * @brief Teste l'alias Point3 pour Vec3
 *
 * Point3 est un alias de type pour Vec3 utilise pour representer
 * les positions dans l'espace 3D. On verifie qu'il se comporte
 * exactement comme un Vec3.
 */
bool test_point3_alias() {
    Point3 p(1.0f, 2.0f, 3.0f);
    TEST_ASSERT_NEAR(p.x, 1.0f, EPS);
    TEST_ASSERT_NEAR(p.y, 2.0f, EPS);
    TEST_ASSERT_NEAR(p.z, 3.0f, EPS);
    return true;
}

/**
 * @brief Teste l'alias Color pour Vec3
 *
 * Color est un alias de type pour Vec3 utilise pour representer
 * les couleurs RGB. On verifie qu'il se comporte comme un Vec3.
 */
bool test_color_alias() {
    Color c(0.5f, 0.3f, 0.1f);
    TEST_ASSERT_NEAR(c.x, 0.5f, EPS);
    TEST_ASSERT_NEAR(c.y, 0.3f, EPS);
    TEST_ASSERT_NEAR(c.z, 0.1f, EPS);
    return true;
}


/**
 * @brief Execute l'ensemble des tests unitaires pour Vec3
 *
 * Fonction principale qui lance tous les tests de Vec3 et met a jour
 * les compteurs de resultats (passed, failed, total).
 *
 * @param passed Compteur de tests reussis
 * @param failed Compteur de tests echoues
 * @param total Compteur du nombre total de tests executes
 */
void run_vec3_tests(int& passed, int& failed, int& total) {
    RUN_TEST(test_vec3_default_constructor);
    RUN_TEST(test_vec3_value_constructor);
    RUN_TEST(test_vec3_single_value_constructor);

    RUN_TEST(test_vec3_negation);
    RUN_TEST(test_vec3_addition);
    RUN_TEST(test_vec3_subtraction);
    RUN_TEST(test_vec3_multiplication);
    RUN_TEST(test_vec3_scalar_multiplication);
    RUN_TEST(test_vec3_division);
    RUN_TEST(test_vec3_index_operator);
    RUN_TEST(test_vec3_compound_addition);
    RUN_TEST(test_vec3_compound_multiplication);
    RUN_TEST(test_vec3_compound_division);

    RUN_TEST(test_vec3_length);
    RUN_TEST(test_vec3_length_squared);
    RUN_TEST(test_vec3_normalized);
    RUN_TEST(test_vec3_near_zero);

    RUN_TEST(test_vec3_dot);
    RUN_TEST(test_vec3_cross);
    RUN_TEST(test_vec3_cross_anticommutative);
    RUN_TEST(test_vec3_reflect);
    RUN_TEST(test_vec3_refract_perpendicular);
    RUN_TEST(test_vec3_unit_vector);

    RUN_TEST(test_point3_alias);
    RUN_TEST(test_color_alias);
}
