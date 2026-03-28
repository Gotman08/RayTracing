/** @file test_bvh_builder.cpp
 * @brief Tests unitaires du BVHBuilder (construction, traversee, free) */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <cfloat>

#include "raytracer/acceleration/bvh_builder.cuh"
#include "raytracer/acceleration/bvh.cuh"
#include "raytracer/geometry/hittable_list.cuh"
#include "raytracer/materials/material.cuh"

using namespace rt;

/** @brief Fixture BVHBuilder : materiau partage + helper make_sphere */
class BVHBuilderTest : public ::testing::Test {
protected:
    Material lambertian_mat;
    BVHBuilder builder;

    void SetUp() override {
        lambertian_mat = Material(MaterialType::LAMBERTIAN, Color(0.5f, 0.5f, 0.5f));
    }

    /** @brief Cree une sphere pour les tests */
    HittableObject make_sphere(const Point3& center, float radius = 1.0f) {
        HittableObject obj;
        obj.type = HittableType::SPHERE;
        obj.data.sphere = Sphere(center, radius, &lambertian_mat);
        obj.bbox = obj.data.sphere.bounding_box();
        return obj;
    }
};

/**
 * @brief Construire un BVH vide ne doit pas crasher et laisser les vecteurs vides
 */
TEST_F(BVHBuilderTest, BuildEmpty) {
    builder.build(nullptr, 0);

    EXPECT_TRUE(builder.nodes.empty());
    EXPECT_TRUE(builder.primitives.empty());
}

/**
 * @brief Un seul objet doit produire exactement 1 noeud feuille
 */
TEST_F(BVHBuilderTest, BuildSingle) {
    HittableObject obj = make_sphere(Point3(0, 0, 0));

    builder.build(&obj, 1);

    ASSERT_EQ(builder.nodes.size(), 1u);
    ASSERT_EQ(builder.primitives.size(), 1u);

    const BVHNode& node = builder.nodes[0];
    EXPECT_TRUE(node.is_leaf);
    EXPECT_EQ(node.primitive_idx, 0);

    EXPECT_NEAR(node.bounds.x.min, -1.0f, 0.01f);
    EXPECT_NEAR(node.bounds.x.max,  1.0f, 0.01f);
    EXPECT_NEAR(node.bounds.y.min, -1.0f, 0.01f);
    EXPECT_NEAR(node.bounds.y.max,  1.0f, 0.01f);
    EXPECT_NEAR(node.bounds.z.min, -1.0f, 0.01f);
    EXPECT_NEAR(node.bounds.z.max,  1.0f, 0.01f);
}

/**
 * @brief Deux spheres separees doivent produire 3 noeuds (1 racine + 2 feuilles)
 */
TEST_F(BVHBuilderTest, BuildTwo) {
    std::vector<HittableObject> objs = {
        make_sphere(Point3(-5, 0, 0)),
        make_sphere(Point3( 5, 0, 0))
    };

    builder.build(objs.data(), static_cast<int>(objs.size()));

    ASSERT_EQ(builder.nodes.size(), 3u);

    const BVHNode& root = builder.nodes[0];
    EXPECT_FALSE(root.is_leaf);

    EXPECT_TRUE(builder.nodes[root.left].is_leaf);
    EXPECT_TRUE(builder.nodes[root.right].is_leaf);
}

/**
 * @brief 5 spheres sur l'axe X : la bbox racine doit toutes les englober
 */
TEST_F(BVHBuilderTest, BuildMultiSorting) {
    std::vector<HittableObject> objs;
    float positions[] = {-10.0f, -3.0f, 0.0f, 4.0f, 12.0f};
    for (float px : positions) {
        objs.push_back(make_sphere(Point3(px, 0, 0)));
    }

    builder.build(objs.data(), static_cast<int>(objs.size()));

    EXPECT_GE(builder.nodes.size(), 5u);

    const AABB& root_box = builder.nodes[0].bounds;
    EXPECT_LE(root_box.x.min, -10.0f + 1.0f);
    EXPECT_GE(root_box.x.max,  12.0f - 1.0f);

    EXPECT_LE(root_box.x.min, -11.0f + 0.01f);
    EXPECT_GE(root_box.x.max,  13.0f - 0.01f);
}

/**
 * @brief La bbox de la racine doit englober la bbox de chaque primitive
 */
TEST_F(BVHBuilderTest, BboxCorrectness) {
    std::vector<HittableObject> objs = {
        make_sphere(Point3(-3, 2, 1)),
        make_sphere(Point3( 4, -1, 5)),
        make_sphere(Point3( 0, 7, -2))
    };

    builder.build(objs.data(), static_cast<int>(objs.size()));

    const AABB& root_box = builder.nodes[0].bounds;

    for (const auto& prim : builder.primitives) {
        EXPECT_LE(root_box.x.min, prim.bbox.x.min + 1e-5f);
        EXPECT_GE(root_box.x.max, prim.bbox.x.max - 1e-5f);
        EXPECT_LE(root_box.y.min, prim.bbox.y.min + 1e-5f);
        EXPECT_GE(root_box.y.max, prim.bbox.y.max - 1e-5f);
        EXPECT_LE(root_box.z.min, prim.bbox.z.min + 1e-5f);
        EXPECT_GE(root_box.z.max, prim.bbox.z.max - 1e-5f);
    }
}

/**
 * @brief Tous les noeuds internes ont des indices enfants valides,
 *        et toutes les feuilles ont un primitive_idx >= 0
 */
TEST_F(BVHBuilderTest, NodeRelationships) {
    std::vector<HittableObject> objs = {
        make_sphere(Point3(-4, 0, 0)),
        make_sphere(Point3(-1, 0, 0)),
        make_sphere(Point3( 2, 0, 0)),
        make_sphere(Point3( 5, 0, 0))
    };

    builder.build(objs.data(), static_cast<int>(objs.size()));

    int num_nodes = static_cast<int>(builder.nodes.size());

    for (int i = 0; i < num_nodes; i++) {
        const BVHNode& node = builder.nodes[i];
        if (node.is_leaf) {
            EXPECT_GE(node.primitive_idx, 0);
            EXPECT_LT(node.primitive_idx, static_cast<int>(builder.primitives.size()));
        } else {
            EXPECT_GE(node.left, 0);
            EXPECT_LT(node.left, num_nodes);
            EXPECT_GE(node.right, 0);
            EXPECT_LT(node.right, num_nodes);
        }
    }
}

/**
 * @brief Construire un BVH CPU et lancer un rayon vers une sphere connue
 */
TEST_F(BVHBuilderTest, CreateCpuBvh) {
    std::vector<HittableObject> objs = {
        make_sphere(Point3(0, 0, -5)),
        make_sphere(Point3(10, 0, -5)),
        make_sphere(Point3(-10, 0, -5))
    };

    builder.build(objs.data(), static_cast<int>(objs.size()));
    BVH bvh = builder.create_cpu_bvh();

    Ray ray(Point3(0, 0, 0), Vec3(0, 0, -1));
    HitRecord rec;
    bool did_hit = bvh.hit(ray, Interval(0.001f, FLT_MAX), rec);

    EXPECT_TRUE(did_hit);
    EXPECT_NEAR(rec.t, 4.0f, 0.1f);

    builder.free_cpu_bvh(bvh);
}

/**
 * @brief Deux spheres sur l'axe Z : le BVH doit retourner la plus proche
 */
TEST_F(BVHBuilderTest, CreateCpuBvhClosestHit) {
    std::vector<HittableObject> objs = {
        make_sphere(Point3(0, 0, -3)),
        make_sphere(Point3(0, 0, -6))
    };

    builder.build(objs.data(), static_cast<int>(objs.size()));
    BVH bvh = builder.create_cpu_bvh();

    Ray ray(Point3(0, 0, 0), Vec3(0, 0, -1));
    HitRecord rec;
    bool did_hit = bvh.hit(ray, Interval(0.001f, FLT_MAX), rec);

    EXPECT_TRUE(did_hit);
    EXPECT_NEAR(rec.t, 2.0f, 0.1f);

    EXPECT_NEAR(rec.p.z, -2.0f, 0.1f);

    builder.free_cpu_bvh(bvh);
}

/**
 * @brief Apres free_cpu_bvh(), les pointeurs nodes et primitives sont nullptr
 */
TEST_F(BVHBuilderTest, FreeCpuBvhNullsPointers) {
    std::vector<HittableObject> objs = {
        make_sphere(Point3(0, 0, 0)),
        make_sphere(Point3(3, 0, 0))
    };

    builder.build(objs.data(), static_cast<int>(objs.size()));
    BVH bvh = builder.create_cpu_bvh();

    EXPECT_NE(bvh.nodes, nullptr);
    EXPECT_NE(bvh.primitives, nullptr);

    builder.free_cpu_bvh(bvh);

    EXPECT_EQ(bvh.nodes, nullptr);
    EXPECT_EQ(bvh.primitives, nullptr);
}
