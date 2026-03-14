/**
 * Unit Tests - BVH (Bounding Volume Hierarchy)
 * Tests BVH node structure and traversal (host-only, no CUDA dependencies)
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

#define TEST_ASSERT(expr) \
    do { \
        if (!(expr)) { \
            std::cerr << "    FAILED: " << #expr << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            return false; \
        } \
    } while(0)

#define TEST_ASSERT_NEAR(a, b, eps) \
    do { \
        if (std::abs((a) - (b)) > (eps)) { \
            std::cerr << "    FAILED: " << #a << " (" << (a) << ") != " << #b << " (" << (b) << ")" \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            return false; \
        } \
    } while(0)

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

// ============================================================================
// BVHNode tests
// ============================================================================

static bool test_bvh_node_default() {
    BVHNode node;
    TEST_ASSERT(node.left == -1);
    TEST_ASSERT(node.right == -1);
    TEST_ASSERT(node.primitive_idx == -1);
    TEST_ASSERT(node.is_leaf == false);
    return true;
}

// ============================================================================
// BVH empty tests
// ============================================================================

static bool test_bvh_empty() {
    BVH bvh;
    TEST_ASSERT(bvh.nodes == nullptr);
    TEST_ASSERT(bvh.primitives == nullptr);
    TEST_ASSERT(bvh.num_nodes == 0);
    TEST_ASSERT(bvh.num_primitives == 0);
    return true;
}

static bool test_bvh_empty_no_hit() {
    BVH bvh;
    Ray r(Point3(0, 0, 0), Vec3(0, 0, -1));
    HitRecord rec;
    bool hit = bvh.hit(r, Interval(0.001f, 1e30f), rec);
    TEST_ASSERT(hit == false);
    return true;
}

// ============================================================================
// BVH manually built tree tests
// ============================================================================

// Helper: build a BVH with a single leaf containing a sphere
static bool test_bvh_single_leaf_hit() {
    Material mat;
    mat.type = MaterialType::LAMBERTIAN;
    mat.albedo = Color(0.5f, 0.5f, 0.5f);

    // Create a sphere object at (0, 0, -3) radius 1
    HittableObject obj;
    obj.type = HittableType::SPHERE;
    obj.data.sphere = Sphere(Point3(0, 0, -3), 1.0f, &mat);
    obj.bbox = obj.data.sphere.bounding_box();

    // Create a single-leaf BVH manually
    BVHNode node;
    node.bounds = obj.bbox;
    node.is_leaf = true;
    node.primitive_idx = 0;

    BVH bvh;
    bvh.nodes = &node;
    bvh.primitives = &obj;
    bvh.num_nodes = 1;
    bvh.num_primitives = 1;

    // Ray hitting the sphere
    Ray r(Point3(0, 0, 0), Vec3(0, 0, -1));
    HitRecord rec;
    bool hit = bvh.hit(r, Interval(0.001f, 1e30f), rec);
    TEST_ASSERT(hit == true);
    TEST_ASSERT_NEAR(rec.t, 2.0f, 1e-4f);

    return true;
}

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

    // Ray missing the sphere (going up)
    Ray r(Point3(0, 0, 0), Vec3(0, 1, 0));
    HitRecord rec;
    bool hit = bvh.hit(r, Interval(0.001f, 1e30f), rec);
    TEST_ASSERT(hit == false);

    return true;
}

static bool test_bvh_two_node_tree() {
    Material mat1, mat2;
    mat1.albedo = Color(1, 0, 0);
    mat2.albedo = Color(0, 1, 0);

    // Two spheres
    HittableObject objs[2];
    objs[0].type = HittableType::SPHERE;
    objs[0].data.sphere = Sphere(Point3(-2, 0, -3), 1.0f, &mat1);
    objs[0].bbox = objs[0].data.sphere.bounding_box();

    objs[1].type = HittableType::SPHERE;
    objs[1].data.sphere = Sphere(Point3(2, 0, -3), 1.0f, &mat2);
    objs[1].bbox = objs[1].data.sphere.bounding_box();

    // Build a 3-node BVH: root -> left leaf, right leaf
    BVHNode nodes[3];

    // Left leaf
    nodes[1].bounds = objs[0].bbox;
    nodes[1].is_leaf = true;
    nodes[1].primitive_idx = 0;

    // Right leaf
    nodes[2].bounds = objs[1].bbox;
    nodes[2].is_leaf = true;
    nodes[2].primitive_idx = 1;

    // Root (internal)
    nodes[0].bounds = AABB(objs[0].bbox, objs[1].bbox);
    nodes[0].is_leaf = false;
    nodes[0].left = 1;
    nodes[0].right = 2;

    BVH bvh;
    bvh.nodes = nodes;
    bvh.primitives = objs;
    bvh.num_nodes = 3;
    bvh.num_primitives = 2;

    // Ray toward left sphere
    Ray r1(Point3(-2, 0, 0), Vec3(0, 0, -1));
    HitRecord rec;
    bool hit1 = bvh.hit(r1, Interval(0.001f, 1e30f), rec);
    TEST_ASSERT(hit1 == true);
    TEST_ASSERT_NEAR(rec.t, 2.0f, 1e-4f);

    // Ray toward right sphere
    Ray r2(Point3(2, 0, 0), Vec3(0, 0, -1));
    bool hit2 = bvh.hit(r2, Interval(0.001f, 1e30f), rec);
    TEST_ASSERT(hit2 == true);
    TEST_ASSERT_NEAR(rec.t, 2.0f, 1e-4f);

    // Ray between both (miss)
    Ray r3(Point3(0, 5, 0), Vec3(0, 0, -1));
    bool hit3 = bvh.hit(r3, Interval(0.001f, 1e30f), rec);
    TEST_ASSERT(hit3 == false);

    return true;
}

static bool test_bvh_finds_closest_of_two() {
    Material mat1, mat2;

    HittableObject objs[2];
    // Near sphere at z=-2
    objs[0].type = HittableType::SPHERE;
    objs[0].data.sphere = Sphere(Point3(0, 0, -2), 0.5f, &mat1);
    objs[0].bbox = objs[0].data.sphere.bounding_box();

    // Far sphere at z=-5
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
    // Should hit the closer sphere (t ≈ 1.5)
    TEST_ASSERT(rec.t < 2.0f);
    TEST_ASSERT(rec.t > 1.0f);

    return true;
}

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
    // Root should encompass both: x in [-4, 4]
    TEST_ASSERT(bbox.x.min <= -3.9f);
    TEST_ASSERT(bbox.x.max >= 3.9f);

    return true;
}

// ============================================================================
// Test runner
// ============================================================================

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
