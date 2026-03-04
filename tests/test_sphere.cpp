/**
 * Unit Tests - Sphere Intersection
 */

#include <iostream>
#include <cmath>

#include "raytracer/core/ray.cuh"
#include "raytracer/core/interval.cuh"
#include "raytracer/core/hit_record.cuh"
#include "raytracer/geometry/sphere.cuh"

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

constexpr float EPS = 1e-5f;

// ==============================================================================
// Ray-Sphere Intersection Tests
// ==============================================================================

bool test_sphere_ray_through_center() {
    // Sphere at origin, radius 1
    Sphere sphere(Point3(0.0f, 0.0f, 0.0f), 1.0f, nullptr);

    // Ray from z=5 pointing toward origin
    Ray ray(Point3(0.0f, 0.0f, 5.0f), Vec3(0.0f, 0.0f, -1.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = sphere.hit(ray, ray_t, rec);
    TEST_ASSERT(hit == true);

    // Should hit at z=1 (front of sphere), t=4
    TEST_ASSERT_NEAR(rec.t, 4.0f, EPS);
    TEST_ASSERT_NEAR(rec.p.z, 1.0f, EPS);

    return true;
}

bool test_sphere_ray_misses() {
    Sphere sphere(Point3(0.0f, 0.0f, 0.0f), 1.0f, nullptr);

    // Ray parallel to sphere, offset by 2 units
    Ray ray(Point3(2.0f, 0.0f, 5.0f), Vec3(0.0f, 0.0f, -1.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = sphere.hit(ray, ray_t, rec);
    TEST_ASSERT(hit == false);

    return true;
}

bool test_sphere_ray_tangent() {
    Sphere sphere(Point3(0.0f, 0.0f, 0.0f), 1.0f, nullptr);

    // Ray tangent to sphere at (1, 0, 0)
    Ray ray(Point3(1.0f, 0.0f, 5.0f), Vec3(0.0f, 0.0f, -1.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = sphere.hit(ray, ray_t, rec);
    TEST_ASSERT(hit == true);

    // Should hit at z=0
    TEST_ASSERT_NEAR(rec.p.x, 1.0f, EPS);
    TEST_ASSERT_NEAR(rec.p.z, 0.0f, EPS);

    return true;
}

bool test_sphere_ray_from_inside() {
    Sphere sphere(Point3(0.0f, 0.0f, 0.0f), 2.0f, nullptr);

    // Ray starting at center, pointing in +z direction
    Ray ray(Point3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, 1.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = sphere.hit(ray, ray_t, rec);
    TEST_ASSERT(hit == true);

    // Should hit at z=2 (exiting sphere)
    TEST_ASSERT_NEAR(rec.t, 2.0f, EPS);
    TEST_ASSERT_NEAR(rec.p.z, 2.0f, EPS);

    return true;
}

bool test_sphere_ray_behind() {
    Sphere sphere(Point3(0.0f, 0.0f, 0.0f), 1.0f, nullptr);

    // Ray pointing away from sphere
    Ray ray(Point3(0.0f, 0.0f, 5.0f), Vec3(0.0f, 0.0f, 1.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = sphere.hit(ray, ray_t, rec);
    TEST_ASSERT(hit == false);

    return true;
}

bool test_sphere_ray_interval_excludes() {
    Sphere sphere(Point3(0.0f, 0.0f, 0.0f), 1.0f, nullptr);

    // Ray would hit at t=4, but interval is [0, 3]
    Ray ray(Point3(0.0f, 0.0f, 5.0f), Vec3(0.0f, 0.0f, -1.0f));
    Interval ray_t(0.001f, 3.0f);
    HitRecord rec;

    bool hit = sphere.hit(ray, ray_t, rec);
    TEST_ASSERT(hit == false);

    return true;
}

// ==============================================================================
// Normal Tests
// ==============================================================================

bool test_sphere_normal_at_front() {
    Sphere sphere(Point3(0.0f, 0.0f, 0.0f), 1.0f, nullptr);

    // Ray hitting front of sphere
    Ray ray(Point3(0.0f, 0.0f, 5.0f), Vec3(0.0f, 0.0f, -1.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    sphere.hit(ray, ray_t, rec);

    // Normal should point outward (+z direction)
    TEST_ASSERT_NEAR(rec.normal.x, 0.0f, EPS);
    TEST_ASSERT_NEAR(rec.normal.y, 0.0f, EPS);
    TEST_ASSERT_NEAR(rec.normal.z, 1.0f, EPS);
    TEST_ASSERT(rec.front_face == true);

    return true;
}

bool test_sphere_normal_from_inside() {
    Sphere sphere(Point3(0.0f, 0.0f, 0.0f), 2.0f, nullptr);

    // Ray starting inside sphere
    Ray ray(Point3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, 1.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    sphere.hit(ray, ray_t, rec);

    // Normal should be flipped to face ray
    TEST_ASSERT_NEAR(rec.normal.z, -1.0f, EPS);
    TEST_ASSERT(rec.front_face == false);

    return true;
}

bool test_sphere_normal_is_unit() {
    Sphere sphere(Point3(1.0f, 2.0f, 3.0f), 2.5f, nullptr);

    // Diagonal ray
    Ray ray(Point3(10.0f, 10.0f, 10.0f), Vec3(-1.0f, -1.0f, -1.0f).normalized());
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = sphere.hit(ray, ray_t, rec);
    if (!hit) return false;  // Ray might miss, skip test

    // Normal should have length 1
    float len = rec.normal.length();
    TEST_ASSERT_NEAR(len, 1.0f, EPS);

    return true;
}

// ==============================================================================
// Different Sphere Configurations
// ==============================================================================

bool test_sphere_offset_center() {
    // Sphere at (5, 0, 0)
    Sphere sphere(Point3(5.0f, 0.0f, 0.0f), 1.0f, nullptr);

    Ray ray(Point3(5.0f, 0.0f, 5.0f), Vec3(0.0f, 0.0f, -1.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = sphere.hit(ray, ray_t, rec);
    TEST_ASSERT(hit == true);
    TEST_ASSERT_NEAR(rec.p.x, 5.0f, EPS);
    TEST_ASSERT_NEAR(rec.p.z, 1.0f, EPS);

    return true;
}

bool test_sphere_large_radius() {
    Sphere sphere(Point3(0.0f, 0.0f, 0.0f), 100.0f, nullptr);

    Ray ray(Point3(0.0f, 0.0f, 200.0f), Vec3(0.0f, 0.0f, -1.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = sphere.hit(ray, ray_t, rec);
    TEST_ASSERT(hit == true);
    TEST_ASSERT_NEAR(rec.t, 100.0f, EPS);

    return true;
}

bool test_sphere_small_radius() {
    Sphere sphere(Point3(0.0f, 0.0f, 0.0f), 0.01f, nullptr);

    Ray ray(Point3(0.0f, 0.0f, 1.0f), Vec3(0.0f, 0.0f, -1.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = sphere.hit(ray, ray_t, rec);
    TEST_ASSERT(hit == true);
    TEST_ASSERT_NEAR(rec.p.z, 0.01f, EPS);

    return true;
}

// ==============================================================================
// Test Suite Runner
// ==============================================================================

void run_sphere_tests(int& passed, int& failed, int& total) {
    // Ray-Sphere Intersection
    RUN_TEST(test_sphere_ray_through_center);
    RUN_TEST(test_sphere_ray_misses);
    RUN_TEST(test_sphere_ray_tangent);
    RUN_TEST(test_sphere_ray_from_inside);
    RUN_TEST(test_sphere_ray_behind);
    RUN_TEST(test_sphere_ray_interval_excludes);

    // Normals
    RUN_TEST(test_sphere_normal_at_front);
    RUN_TEST(test_sphere_normal_from_inside);
    RUN_TEST(test_sphere_normal_is_unit);

    // Different configurations
    RUN_TEST(test_sphere_offset_center);
    RUN_TEST(test_sphere_large_radius);
    RUN_TEST(test_sphere_small_radius);
}
