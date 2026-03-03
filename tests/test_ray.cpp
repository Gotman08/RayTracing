/**
 * Unit Tests - Ray
 */

#include <iostream>
#include <cmath>

#include "raytracer/core/ray.cuh"

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

constexpr float EPS = 1e-6f;

// ==============================================================================
// Construction Tests
// ==============================================================================

bool test_ray_default_constructor() {
    Ray r;
    TEST_ASSERT_NEAR(r.origin().x, 0.0f, EPS);
    TEST_ASSERT_NEAR(r.origin().y, 0.0f, EPS);
    TEST_ASSERT_NEAR(r.origin().z, 0.0f, EPS);
    TEST_ASSERT_NEAR(r.direction().x, 0.0f, EPS);
    TEST_ASSERT_NEAR(r.direction().y, 0.0f, EPS);
    TEST_ASSERT_NEAR(r.direction().z, 0.0f, EPS);
    TEST_ASSERT_NEAR(r.time(), 0.0f, EPS);
    return true;
}

bool test_ray_parameterized_constructor() {
    Point3 origin(1.0f, 2.0f, 3.0f);
    Vec3 direction(0.0f, 0.0f, -1.0f);
    Ray r(origin, direction);

    TEST_ASSERT_NEAR(r.origin().x, 1.0f, EPS);
    TEST_ASSERT_NEAR(r.origin().y, 2.0f, EPS);
    TEST_ASSERT_NEAR(r.origin().z, 3.0f, EPS);
    TEST_ASSERT_NEAR(r.direction().x, 0.0f, EPS);
    TEST_ASSERT_NEAR(r.direction().y, 0.0f, EPS);
    TEST_ASSERT_NEAR(r.direction().z, -1.0f, EPS);
    TEST_ASSERT_NEAR(r.time(), 0.0f, EPS);
    return true;
}

bool test_ray_with_time() {
    Point3 origin(0.0f, 0.0f, 0.0f);
    Vec3 direction(1.0f, 0.0f, 0.0f);
    Ray r(origin, direction, 0.5f);

    TEST_ASSERT_NEAR(r.time(), 0.5f, EPS);
    return true;
}

// ==============================================================================
// at(t) Tests - Verify p = o + t*d formula
// ==============================================================================

bool test_ray_at_zero() {
    Point3 origin(1.0f, 2.0f, 3.0f);
    Vec3 direction(1.0f, 0.0f, 0.0f);
    Ray r(origin, direction);

    Point3 p = r.at(0.0f);
    // At t=0, p should equal origin
    TEST_ASSERT_NEAR(p.x, 1.0f, EPS);
    TEST_ASSERT_NEAR(p.y, 2.0f, EPS);
    TEST_ASSERT_NEAR(p.z, 3.0f, EPS);
    return true;
}

bool test_ray_at_positive() {
    Point3 origin(0.0f, 0.0f, 0.0f);
    Vec3 direction(1.0f, 2.0f, 3.0f);
    Ray r(origin, direction);

    Point3 p = r.at(2.0f);
    // p = (0,0,0) + 2*(1,2,3) = (2,4,6)
    TEST_ASSERT_NEAR(p.x, 2.0f, EPS);
    TEST_ASSERT_NEAR(p.y, 4.0f, EPS);
    TEST_ASSERT_NEAR(p.z, 6.0f, EPS);
    return true;
}

bool test_ray_at_negative() {
    Point3 origin(5.0f, 5.0f, 5.0f);
    Vec3 direction(1.0f, 0.0f, 0.0f);
    Ray r(origin, direction);

    Point3 p = r.at(-3.0f);
    // p = (5,5,5) + (-3)*(1,0,0) = (2,5,5)
    TEST_ASSERT_NEAR(p.x, 2.0f, EPS);
    TEST_ASSERT_NEAR(p.y, 5.0f, EPS);
    TEST_ASSERT_NEAR(p.z, 5.0f, EPS);
    return true;
}

bool test_ray_at_fractional() {
    Point3 origin(0.0f, 0.0f, 0.0f);
    Vec3 direction(4.0f, 0.0f, 0.0f);
    Ray r(origin, direction);

    Point3 p = r.at(0.5f);
    // p = (0,0,0) + 0.5*(4,0,0) = (2,0,0)
    TEST_ASSERT_NEAR(p.x, 2.0f, EPS);
    TEST_ASSERT_NEAR(p.y, 0.0f, EPS);
    TEST_ASSERT_NEAR(p.z, 0.0f, EPS);
    return true;
}

bool test_ray_at_unit_direction() {
    Point3 origin(0.0f, 0.0f, 0.0f);
    Vec3 direction = Vec3(1.0f, 1.0f, 1.0f).normalized();
    Ray r(origin, direction);

    Point3 p = r.at(1.0f);
    // With unit direction, at t=1, distance from origin should be 1
    float dist = (p - origin).length();
    TEST_ASSERT_NEAR(dist, 1.0f, EPS);
    return true;
}

// ==============================================================================
// Ray Direction Tests
// ==============================================================================

bool test_ray_diagonal_direction() {
    Point3 origin(1.0f, 1.0f, 1.0f);
    Vec3 direction(1.0f, 1.0f, 1.0f);
    Ray r(origin, direction);

    Point3 p = r.at(3.0f);
    // p = (1,1,1) + 3*(1,1,1) = (4,4,4)
    TEST_ASSERT_NEAR(p.x, 4.0f, EPS);
    TEST_ASSERT_NEAR(p.y, 4.0f, EPS);
    TEST_ASSERT_NEAR(p.z, 4.0f, EPS);
    return true;
}

bool test_ray_z_direction() {
    Point3 origin(0.0f, 0.0f, 0.0f);
    Vec3 direction(0.0f, 0.0f, -1.0f);  // Looking into scene
    Ray r(origin, direction);

    Point3 p = r.at(10.0f);
    // p = (0,0,0) + 10*(0,0,-1) = (0,0,-10)
    TEST_ASSERT_NEAR(p.x, 0.0f, EPS);
    TEST_ASSERT_NEAR(p.y, 0.0f, EPS);
    TEST_ASSERT_NEAR(p.z, -10.0f, EPS);
    return true;
}

// ==============================================================================
// Test Suite Runner
// ==============================================================================

void run_ray_tests(int& passed, int& failed, int& total) {
    // Construction
    RUN_TEST(test_ray_default_constructor);
    RUN_TEST(test_ray_parameterized_constructor);
    RUN_TEST(test_ray_with_time);

    // at(t) - formula p = o + t*d
    RUN_TEST(test_ray_at_zero);
    RUN_TEST(test_ray_at_positive);
    RUN_TEST(test_ray_at_negative);
    RUN_TEST(test_ray_at_fractional);
    RUN_TEST(test_ray_at_unit_direction);

    // Direction variations
    RUN_TEST(test_ray_diagonal_direction);
    RUN_TEST(test_ray_z_direction);
}
