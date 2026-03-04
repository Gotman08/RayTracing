/**
 * Unit Tests - AABB (Axis-Aligned Bounding Box)
 */

#include <iostream>
#include <cmath>

#include "raytracer/core/aabb.cuh"
#include "raytracer/core/cuda_utils.cuh"

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

constexpr float EPS = 1e-4f;  // Slightly larger for AABB due to padding

// ==============================================================================
// Construction Tests
// ==============================================================================

bool test_aabb_point_constructor() {
    Point3 a(0.0f, 0.0f, 0.0f);
    Point3 b(2.0f, 3.0f, 4.0f);
    AABB box(a, b);

    TEST_ASSERT_NEAR(box.x.min, 0.0f, EPS);
    TEST_ASSERT_NEAR(box.x.max, 2.0f, EPS);
    TEST_ASSERT_NEAR(box.y.min, 0.0f, EPS);
    TEST_ASSERT_NEAR(box.y.max, 3.0f, EPS);
    TEST_ASSERT_NEAR(box.z.min, 0.0f, EPS);
    TEST_ASSERT_NEAR(box.z.max, 4.0f, EPS);

    return true;
}

bool test_aabb_point_constructor_reversed() {
    // Points in wrong order should still create correct box
    Point3 a(5.0f, 5.0f, 5.0f);
    Point3 b(0.0f, 0.0f, 0.0f);
    AABB box(a, b);

    TEST_ASSERT_NEAR(box.x.min, 0.0f, EPS);
    TEST_ASSERT_NEAR(box.x.max, 5.0f, EPS);

    return true;
}

bool test_aabb_interval_constructor() {
    Interval ix(0.0f, 2.0f);
    Interval iy(0.0f, 3.0f);
    Interval iz(0.0f, 4.0f);
    AABB box(ix, iy, iz);

    TEST_ASSERT_NEAR(box.x.min, 0.0f, EPS);
    TEST_ASSERT_NEAR(box.x.max, 2.0f, EPS);
    TEST_ASSERT_NEAR(box.y.min, 0.0f, EPS);
    TEST_ASSERT_NEAR(box.y.max, 3.0f, EPS);
    TEST_ASSERT_NEAR(box.z.min, 0.0f, EPS);
    TEST_ASSERT_NEAR(box.z.max, 4.0f, EPS);

    return true;
}

bool test_aabb_merge_constructor() {
    AABB box1(Point3(0.0f, 0.0f, 0.0f), Point3(1.0f, 1.0f, 1.0f));
    AABB box2(Point3(2.0f, 2.0f, 2.0f), Point3(3.0f, 3.0f, 3.0f));
    AABB merged(box1, box2);

    TEST_ASSERT_NEAR(merged.x.min, 0.0f, EPS);
    TEST_ASSERT_NEAR(merged.x.max, 3.0f, EPS);
    TEST_ASSERT_NEAR(merged.y.min, 0.0f, EPS);
    TEST_ASSERT_NEAR(merged.y.max, 3.0f, EPS);

    return true;
}

// ==============================================================================
// Hit Tests
// ==============================================================================

bool test_aabb_hit_through_center() {
    AABB box(Point3(0.0f, 0.0f, 0.0f), Point3(2.0f, 2.0f, 2.0f));

    // Ray through center
    Ray ray(Point3(1.0f, 1.0f, -5.0f), Vec3(0.0f, 0.0f, 1.0f));
    Interval ray_t(0.001f, INFINITY_F);

    bool hit = box.hit(ray, ray_t);
    TEST_ASSERT(hit == true);

    return true;
}

bool test_aabb_hit_edge() {
    AABB box(Point3(0.0f, 0.0f, 0.0f), Point3(2.0f, 2.0f, 2.0f));

    // Ray slightly inside edge (exact edge is a degenerate case)
    Ray ray(Point3(0.001f, 0.001f, -5.0f), Vec3(0.0f, 0.0f, 1.0f));
    Interval ray_t(0.001f, INFINITY_F);

    bool hit = box.hit(ray, ray_t);
    TEST_ASSERT(hit == true);

    return true;
}

bool test_aabb_miss() {
    AABB box(Point3(0.0f, 0.0f, 0.0f), Point3(2.0f, 2.0f, 2.0f));

    // Ray missing box
    Ray ray(Point3(5.0f, 5.0f, -5.0f), Vec3(0.0f, 0.0f, 1.0f));
    Interval ray_t(0.001f, INFINITY_F);

    bool hit = box.hit(ray, ray_t);
    TEST_ASSERT(hit == false);

    return true;
}

bool test_aabb_ray_behind() {
    AABB box(Point3(0.0f, 0.0f, 0.0f), Point3(2.0f, 2.0f, 2.0f));

    // Ray pointing away from box
    Ray ray(Point3(1.0f, 1.0f, -5.0f), Vec3(0.0f, 0.0f, -1.0f));
    Interval ray_t(0.001f, INFINITY_F);

    bool hit = box.hit(ray, ray_t);
    TEST_ASSERT(hit == false);

    return true;
}

bool test_aabb_ray_inside() {
    AABB box(Point3(0.0f, 0.0f, 0.0f), Point3(10.0f, 10.0f, 10.0f));

    // Ray starting inside box
    Ray ray(Point3(5.0f, 5.0f, 5.0f), Vec3(1.0f, 0.0f, 0.0f));
    Interval ray_t(0.001f, INFINITY_F);

    bool hit = box.hit(ray, ray_t);
    TEST_ASSERT(hit == true);

    return true;
}

bool test_aabb_diagonal_ray() {
    AABB box(Point3(0.0f, 0.0f, 0.0f), Point3(2.0f, 2.0f, 2.0f));

    // Diagonal ray through box
    Ray ray(Point3(-1.0f, -1.0f, -1.0f), Vec3(1.0f, 1.0f, 1.0f).normalized());
    Interval ray_t(0.001f, INFINITY_F);

    bool hit = box.hit(ray, ray_t);
    TEST_ASSERT(hit == true);

    return true;
}

bool test_aabb_interval_excludes() {
    AABB box(Point3(0.0f, 0.0f, 0.0f), Point3(2.0f, 2.0f, 2.0f));

    // Ray would hit, but interval too short
    Ray ray(Point3(1.0f, 1.0f, -10.0f), Vec3(0.0f, 0.0f, 1.0f));
    Interval ray_t(0.001f, 5.0f);  // Box is at z >= 0, ray starts at z=-10

    bool hit = box.hit(ray, ray_t);
    TEST_ASSERT(hit == false);

    return true;
}

// ==============================================================================
// Centroid Tests
// ==============================================================================

bool test_aabb_centroid() {
    AABB box(Point3(0.0f, 0.0f, 0.0f), Point3(4.0f, 6.0f, 8.0f));
    Point3 c = box.centroid();

    TEST_ASSERT_NEAR(c.x, 2.0f, EPS);
    TEST_ASSERT_NEAR(c.y, 3.0f, EPS);
    TEST_ASSERT_NEAR(c.z, 4.0f, EPS);

    return true;
}

bool test_aabb_centroid_offset() {
    AABB box(Point3(2.0f, 2.0f, 2.0f), Point3(4.0f, 4.0f, 4.0f));
    Point3 c = box.centroid();

    TEST_ASSERT_NEAR(c.x, 3.0f, EPS);
    TEST_ASSERT_NEAR(c.y, 3.0f, EPS);
    TEST_ASSERT_NEAR(c.z, 3.0f, EPS);

    return true;
}

// ==============================================================================
// Surface Area Tests
// ==============================================================================

bool test_aabb_surface_area_cube() {
    // 2x2x2 cube
    AABB box(Point3(0.0f, 0.0f, 0.0f), Point3(2.0f, 2.0f, 2.0f));
    float sa = box.surface_area();

    // Surface area = 2 * (2*2 + 2*2 + 2*2) = 2 * 12 = 24
    TEST_ASSERT_NEAR(sa, 24.0f, EPS);

    return true;
}

bool test_aabb_surface_area_rect() {
    // 2x3x4 box
    AABB box(Point3(0.0f, 0.0f, 0.0f), Point3(2.0f, 3.0f, 4.0f));
    float sa = box.surface_area();

    // Surface area = 2 * (2*3 + 3*4 + 4*2) = 2 * (6 + 12 + 8) = 52
    TEST_ASSERT_NEAR(sa, 52.0f, EPS);

    return true;
}

// ==============================================================================
// Longest Axis Tests
// ==============================================================================

bool test_aabb_longest_axis_x() {
    AABB box(Point3(0.0f, 0.0f, 0.0f), Point3(10.0f, 2.0f, 3.0f));
    TEST_ASSERT(box.longest_axis() == 0);  // x is longest

    return true;
}

bool test_aabb_longest_axis_y() {
    AABB box(Point3(0.0f, 0.0f, 0.0f), Point3(2.0f, 10.0f, 3.0f));
    TEST_ASSERT(box.longest_axis() == 1);  // y is longest

    return true;
}

bool test_aabb_longest_axis_z() {
    AABB box(Point3(0.0f, 0.0f, 0.0f), Point3(2.0f, 3.0f, 10.0f));
    TEST_ASSERT(box.longest_axis() == 2);  // z is longest

    return true;
}

// ==============================================================================
// Axis Interval Tests
// ==============================================================================

bool test_aabb_axis_interval() {
    AABB box(Point3(1.0f, 2.0f, 3.0f), Point3(4.0f, 5.0f, 6.0f));

    const Interval& ix = box.axis_interval(0);
    const Interval& iy = box.axis_interval(1);
    const Interval& iz = box.axis_interval(2);

    TEST_ASSERT_NEAR(ix.min, 1.0f, EPS);
    TEST_ASSERT_NEAR(ix.max, 4.0f, EPS);
    TEST_ASSERT_NEAR(iy.min, 2.0f, EPS);
    TEST_ASSERT_NEAR(iy.max, 5.0f, EPS);
    TEST_ASSERT_NEAR(iz.min, 3.0f, EPS);
    TEST_ASSERT_NEAR(iz.max, 6.0f, EPS);

    return true;
}

// ==============================================================================
// Test Suite Runner
// ==============================================================================

void run_aabb_tests(int& passed, int& failed, int& total) {
    // Construction
    RUN_TEST(test_aabb_point_constructor);
    RUN_TEST(test_aabb_point_constructor_reversed);
    RUN_TEST(test_aabb_interval_constructor);
    RUN_TEST(test_aabb_merge_constructor);

    // Hit
    RUN_TEST(test_aabb_hit_through_center);
    RUN_TEST(test_aabb_hit_edge);
    RUN_TEST(test_aabb_miss);
    RUN_TEST(test_aabb_ray_behind);
    RUN_TEST(test_aabb_ray_inside);
    RUN_TEST(test_aabb_diagonal_ray);
    RUN_TEST(test_aabb_interval_excludes);

    // Centroid
    RUN_TEST(test_aabb_centroid);
    RUN_TEST(test_aabb_centroid_offset);

    // Surface Area
    RUN_TEST(test_aabb_surface_area_cube);
    RUN_TEST(test_aabb_surface_area_rect);

    // Longest Axis
    RUN_TEST(test_aabb_longest_axis_x);
    RUN_TEST(test_aabb_longest_axis_y);
    RUN_TEST(test_aabb_longest_axis_z);

    // Axis Interval
    RUN_TEST(test_aabb_axis_interval);
}
