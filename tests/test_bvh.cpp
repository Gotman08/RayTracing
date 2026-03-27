/**
 * @file test_bvh.cpp
 * @brief Tests BVH : noeuds, traversee, hit/miss, closest, bounding box
 */

#include <gtest/gtest.h>
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

/** @brief BVHNode par defaut : left/right/prim = -1, pas feuille */
TEST(BVHTest, NodeDefault) {
    BVHNode node;
    EXPECT_TRUE(node.left == -1);
    EXPECT_TRUE(node.right == -1);
    EXPECT_TRUE(node.primitive_idx == -1);
    EXPECT_TRUE(node.is_leaf == false);
}


/** @brief BVH vide : num_nodes=0, ptrs null */
TEST(BVHTest, Empty) {
    BVH bvh;
    EXPECT_TRUE(bvh.nodes == nullptr);
    EXPECT_TRUE(bvh.primitives == nullptr);
    EXPECT_TRUE(bvh.num_nodes == 0);
    EXPECT_TRUE(bvh.num_primitives == 0);
}

/** @brief BVH vide : pas de hit */
TEST(BVHTest, EmptyNoHit) {
    BVH bvh;
    Ray r(Point3(0, 0, 0), Vec3(0, 0, -1));
    HitRecord rec;
    bool hit = bvh.hit(r, Interval(0.001f, 1e30f), rec);
    EXPECT_TRUE(hit == false);
}


/** @brief 1 feuille sphere z=-3 -> hit a t=2.0 */
TEST(BVHTest, SingleLeafHit) {
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
    EXPECT_TRUE(hit == true);
    EXPECT_NEAR(rec.t, 2.0f, 1e-4f);
}

/** @brief Rayon vers +Y, sphere en -Z -> miss */
TEST(BVHTest, SingleLeafMiss) {
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
    EXPECT_TRUE(hit == false);
}

/** @brief Arbre 2 feuilles : hit gauche, hit droite, miss ailleurs */
TEST(BVHTest, TwoNodeTree) {
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
    EXPECT_TRUE(hit1 == true);
    EXPECT_NEAR(rec.t, 2.0f, 1e-4f);

    Ray r2(Point3(2, 0, 0), Vec3(0, 0, -1));
    bool hit2 = bvh.hit(r2, Interval(0.001f, 1e30f), rec);
    EXPECT_TRUE(hit2 == true);
    EXPECT_NEAR(rec.t, 2.0f, 1e-4f);

    Ray r3(Point3(0, 5, 0), Vec3(0, 0, -1));
    bool hit3 = bvh.hit(r3, Interval(0.001f, 1e30f), rec);
    EXPECT_TRUE(hit3 == false);
}

/** @brief 2 spheres alignees Z -> retourne la plus proche */
TEST(BVHTest, FindsClosestOfTwo) {
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
    EXPECT_TRUE(hit == true);
    EXPECT_TRUE(rec.t < 2.0f);
    EXPECT_TRUE(rec.t > 1.0f);
}

/** @brief BBox racine englobe les 2 spheres X=-3 et X=+3 */
TEST(BVHTest, BoundingBoxRoot) {
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
    EXPECT_TRUE(bbox.x.min <= -3.9f);
    EXPECT_TRUE(bbox.x.max >= 3.9f);
}
