/**
 * @file test_bvh.cpp
 * @brief Tests unitaires pour la structure d'acceleration BVH (Bounding Volume Hierarchy).
 *
 * Ce fichier teste le bon fonctionnement du BVH : valeurs par defaut des noeuds,
 * comportement a vide, intersection avec une feuille unique, traversee d'un arbre
 * a deux noeuds, recherche de l'intersection la plus proche, et verification
 * de la boite englobante racine.
 */

#include <iostream>
#include <cmath>

#include "raytracer/core/vec3.cuh"
#include "raytracer/core/ray.cuh"
#include "raytracer/core/aabb.cuh"
#include "raytracer/core/interval.cuh"
#include "raytracer/core/hit_record.cuh"
#include "raytracer/materials/material.cuh"
#include "raytracer/geometry/hittable_list.cuh"
#include "raytracer/acceleration/bvh.cuh"

using namespace rt;

/**
 * @brief Macro d'assertion simple pour les tests.
 *
 * Verifie qu'une expression booleenne est vraie. En cas d'echec,
 * affiche le fichier et la ligne puis retourne false.
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
 * Compare deux valeurs en autorisant une marge d'erreur epsilon.
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
 * @brief Verifie les valeurs par defaut d'un noeud BVH a sa creation.
 *
 * Un noeud fraichement cree doit avoir ses indices left, right et primitive_idx
 * a -1, et ne doit pas etre une feuille.
 */
static bool test_bvh_node_default() {
    BVHNode node;
    TEST_ASSERT(node.left == -1);
    TEST_ASSERT(node.right == -1);
    TEST_ASSERT(node.primitive_idx == -1);
    TEST_ASSERT(node.is_leaf == false);
    return true;
}


/**
 * @brief Verifie qu'un BVH vide a ses pointeurs et compteurs a zero.
 */
static bool test_bvh_empty() {
    BVH bvh;
    TEST_ASSERT(bvh.nodes == nullptr);
    TEST_ASSERT(bvh.primitives == nullptr);
    TEST_ASSERT(bvh.num_nodes == 0);
    TEST_ASSERT(bvh.num_primitives == 0);
    return true;
}

/**
 * @brief Verifie qu'un rayon ne touche rien dans un BVH vide.
 */
static bool test_bvh_empty_no_hit() {
    BVH bvh;
    Ray r(Point3(0, 0, 0), Vec3(0, 0, -1));
    HitRecord rec;
    bool hit = bvh.hit(r, Interval(0.001f, 1e30f), rec);
    TEST_ASSERT(hit == false);
    return true;
}


/**
 * @brief Verifie l'intersection correcte d'un rayon avec un BVH a une seule feuille.
 *
 * On place une sphere en (0, 0, -3) de rayon 1, et on tire un rayon depuis
 * l'origine vers -Z. Le rayon doit toucher la sphere a t = 2.0.
 */
static bool test_bvh_single_leaf_hit() {
    Material mat;
    mat.type = MaterialType::LAMBERTIAN;
    mat.albedo = Color(0.5f, 0.5f, 0.5f);

    HittableObject obj;
    obj.type = HittableType::SPHERE;
    obj.data.sphere = Sphere(Point3(0, 0, -3), 1.0f, &mat);
    obj.bbox = obj.data.sphere.bounding_box();

    BVHNode node;
    node.bounds = obj.bbox;
    node.is_leaf = true;
    node.primitive_idx = 0;

    BVH bvh;
    bvh.nodes = &node;
    bvh.primitives = &obj;
    bvh.num_nodes = 1;
    bvh.num_primitives = 1;

    Ray r(Point3(0, 0, 0), Vec3(0, 0, -1));
    HitRecord rec;
    bool hit = bvh.hit(r, Interval(0.001f, 1e30f), rec);
    TEST_ASSERT(hit == true);
    TEST_ASSERT_NEAR(rec.t, 2.0f, 1e-4f);

    return true;
}

/**
 * @brief Verifie qu'un rayon qui ne vise pas la sphere rate bien la feuille du BVH.
 *
 * Le rayon part vers +Y alors que la sphere est en -Z : aucune intersection.
 */
static bool test_bvh_single_leaf_miss() {
    Material mat;

    HittableObject obj;
    obj.type = HittableType::SPHERE;
    obj.data.sphere = Sphere(Point3(0, 0, -3), 1.0f, &mat);
    obj.bbox = obj.data.sphere.bounding_box();

    BVHNode node;
    node.bounds = obj.bbox;
    node.is_leaf = true;
    node.primitive_idx = 0;

    BVH bvh;
    bvh.nodes = &node;
    bvh.primitives = &obj;
    bvh.num_nodes = 1;
    bvh.num_primitives = 1;

    Ray r(Point3(0, 0, 0), Vec3(0, 1, 0));
    HitRecord rec;
    bool hit = bvh.hit(r, Interval(0.001f, 1e30f), rec);
    TEST_ASSERT(hit == false);

    return true;
}

/**
 * @brief Verifie la traversee d'un arbre BVH a deux feuilles (noeud racine + 2 enfants).
 *
 * On construit manuellement un arbre avec deux spheres placees symetriquement en X.
 * On teste que chaque sphere est touchee par un rayon dirige vers elle,
 * et qu'un rayon tire ailleurs ne touche rien.
 */
static bool test_bvh_two_node_tree() {
    Material mat1, mat2;
    mat1.albedo = Color(1, 0, 0);
    mat2.albedo = Color(0, 1, 0);

    HittableObject objs[2];
    objs[0].type = HittableType::SPHERE;
    objs[0].data.sphere = Sphere(Point3(-2, 0, -3), 1.0f, &mat1);
    objs[0].bbox = objs[0].data.sphere.bounding_box();

    objs[1].type = HittableType::SPHERE;
    objs[1].data.sphere = Sphere(Point3(2, 0, -3), 1.0f, &mat2);
    objs[1].bbox = objs[1].data.sphere.bounding_box();

    BVHNode nodes[3];

    nodes[1].bounds = objs[0].bbox;
    nodes[1].is_leaf = true;
    nodes[1].primitive_idx = 0;

    nodes[2].bounds = objs[1].bbox;
    nodes[2].is_leaf = true;
    nodes[2].primitive_idx = 1;

    nodes[0].bounds = AABB(objs[0].bbox, objs[1].bbox);
    nodes[0].is_leaf = false;
    nodes[0].left = 1;
    nodes[0].right = 2;

    BVH bvh;
    bvh.nodes = nodes;
    bvh.primitives = objs;
    bvh.num_nodes = 3;
    bvh.num_primitives = 2;

    Ray r1(Point3(-2, 0, 0), Vec3(0, 0, -1));
    HitRecord rec;
    bool hit1 = bvh.hit(r1, Interval(0.001f, 1e30f), rec);
    TEST_ASSERT(hit1 == true);
    TEST_ASSERT_NEAR(rec.t, 2.0f, 1e-4f);

    Ray r2(Point3(2, 0, 0), Vec3(0, 0, -1));
    bool hit2 = bvh.hit(r2, Interval(0.001f, 1e30f), rec);
    TEST_ASSERT(hit2 == true);
    TEST_ASSERT_NEAR(rec.t, 2.0f, 1e-4f);

    Ray r3(Point3(0, 5, 0), Vec3(0, 0, -1));
    bool hit3 = bvh.hit(r3, Interval(0.001f, 1e30f), rec);
    TEST_ASSERT(hit3 == false);

    return true;
}

/**
 * @brief Verifie que le BVH retourne bien l'intersection la plus proche parmi deux objets.
 *
 * Deux spheres alignees sur l'axe Z : une a z=-2 et une a z=-5. Le rayon doit
 * toucher la plus proche (t entre 1 et 2) et non la plus eloignee.
 */
static bool test_bvh_finds_closest_of_two() {
    Material mat1, mat2;

    HittableObject objs[2];
    objs[0].type = HittableType::SPHERE;
    objs[0].data.sphere = Sphere(Point3(0, 0, -2), 0.5f, &mat1);
    objs[0].bbox = objs[0].data.sphere.bounding_box();

    objs[1].type = HittableType::SPHERE;
    objs[1].data.sphere = Sphere(Point3(0, 0, -5), 0.5f, &mat2);
    objs[1].bbox = objs[1].data.sphere.bounding_box();

    BVHNode nodes[3];
    nodes[1].bounds = objs[0].bbox;
    nodes[1].is_leaf = true;
    nodes[1].primitive_idx = 0;

    nodes[2].bounds = objs[1].bbox;
    nodes[2].is_leaf = true;
    nodes[2].primitive_idx = 1;

    nodes[0].bounds = AABB(objs[0].bbox, objs[1].bbox);
    nodes[0].is_leaf = false;
    nodes[0].left = 1;
    nodes[0].right = 2;

    BVH bvh;
    bvh.nodes = nodes;
    bvh.primitives = objs;
    bvh.num_nodes = 3;
    bvh.num_primitives = 2;

    Ray r(Point3(0, 0, 0), Vec3(0, 0, -1));
    HitRecord rec;
    bool hit = bvh.hit(r, Interval(0.001f, 1e30f), rec);
    TEST_ASSERT(hit == true);
    TEST_ASSERT(rec.t < 2.0f);
    TEST_ASSERT(rec.t > 1.0f);

    return true;
}

/**
 * @brief Verifie que la boite englobante racine du BVH contient bien tous les primitifs.
 *
 * Deux spheres eloignees sur l'axe X (-3 et +3). La boite englobante racine
 * doit s'etendre au moins de -3.9 a +3.9 en X (sphere de rayon 1).
 */
static bool test_bvh_bounding_box_root() {
    Material mat;

    HittableObject objs[2];
    objs[0].type = HittableType::SPHERE;
    objs[0].data.sphere = Sphere(Point3(-3, 0, 0), 1.0f, &mat);
    objs[0].bbox = objs[0].data.sphere.bounding_box();

    objs[1].type = HittableType::SPHERE;
    objs[1].data.sphere = Sphere(Point3(3, 0, 0), 1.0f, &mat);
    objs[1].bbox = objs[1].data.sphere.bounding_box();

    BVHNode nodes[3];
    nodes[1].bounds = objs[0].bbox;
    nodes[1].is_leaf = true;
    nodes[1].primitive_idx = 0;
    nodes[2].bounds = objs[1].bbox;
    nodes[2].is_leaf = true;
    nodes[2].primitive_idx = 1;
    nodes[0].bounds = AABB(objs[0].bbox, objs[1].bbox);
    nodes[0].is_leaf = false;
    nodes[0].left = 1;
    nodes[0].right = 2;

    BVH bvh;
    bvh.nodes = nodes;
    bvh.primitives = objs;
    bvh.num_nodes = 3;
    bvh.num_primitives = 2;

    AABB bbox = bvh.bounding_box();
    TEST_ASSERT(bbox.x.min <= -3.9f);
    TEST_ASSERT(bbox.x.max >= 3.9f);

    return true;
}


/**
 * @brief Lance l'ensemble des tests unitaires du BVH.
 *
 * @param passed Compteur de tests reussis (mis a jour par reference).
 * @param failed Compteur de tests echoues (mis a jour par reference).
 * @param total  Compteur total de tests executes (mis a jour par reference).
 */
void run_bvh_tests(int& passed, int& failed, int& total) {
    RUN_TEST(test_bvh_node_default);
    RUN_TEST(test_bvh_empty);
    RUN_TEST(test_bvh_empty_no_hit);
    RUN_TEST(test_bvh_single_leaf_hit);
    RUN_TEST(test_bvh_single_leaf_miss);
    RUN_TEST(test_bvh_two_node_tree);
    RUN_TEST(test_bvh_finds_closest_of_two);
    RUN_TEST(test_bvh_bounding_box_root);
}
