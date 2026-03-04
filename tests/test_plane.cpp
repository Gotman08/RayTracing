/**
 * Unit Tests - Plane Intersection
 */

#include <iostream>
#include <cmath>

#include "raytracer/core/ray.cuh"
#include "raytracer/core/interval.cuh"
#include "raytracer/core/hit_record.cuh"
#include "raytracer/geometry/plane.cuh"

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
// Ray-Plane Intersection Tests
// ==============================================================================

bool test_plane_ray_perpendicular() {
    // Horizontal plane at y=0, normal pointing up
    Plane plane(Point3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), nullptr);

    // Ray from above, pointing straight down
    Ray ray(Point3(0.0f, 5.0f, 0.0f), Vec3(0.0f, -1.0f, 0.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = plane.hit(ray, ray_t, rec);
    TEST_ASSERT(hit == true);

    // Should hit at y=0, t=5
    TEST_ASSERT_NEAR(rec.t, 5.0f, EPS);
    TEST_ASSERT_NEAR(rec.p.y, 0.0f, EPS);

    return true;
}

bool test_plane_ray_oblique() {
    // Horizontal plane at y=0
    Plane plane(Point3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), nullptr);

    // Ray at 45 degrees
    Ray ray(Point3(0.0f, 5.0f, 5.0f), Vec3(0.0f, -1.0f, -1.0f).normalized());
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = plane.hit(ray, ray_t, rec);
    TEST_ASSERT(hit == true);

    // Should hit at y=0
    TEST_ASSERT_NEAR(rec.p.y, 0.0f, EPS);

    return true;
}

bool test_plane_ray_parallel() {
    // Horizontal plane at y=0
    Plane plane(Point3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), nullptr);

    // Ray parallel to plane (in xz direction)
    Ray ray(Point3(0.0f, 5.0f, 0.0f), Vec3(1.0f, 0.0f, 0.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = plane.hit(ray, ray_t, rec);
    TEST_ASSERT(hit == false);

    return true;
}

bool test_plane_ray_pointing_away() {
    // Horizontal plane at y=0
    Plane plane(Point3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), nullptr);

    // Ray pointing away from plane
    Ray ray(Point3(0.0f, 5.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = plane.hit(ray, ray_t, rec);
    TEST_ASSERT(hit == false);

    return true;
}

bool test_plane_ray_from_behind() {
    // Horizontal plane at y=0, normal pointing up
    Plane plane(Point3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), nullptr);

    // Ray from below, hitting back of plane
    Ray ray(Point3(0.0f, -5.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = plane.hit(ray, ray_t, rec);
    TEST_ASSERT(hit == true);

    // Should hit, back face
    TEST_ASSERT(rec.front_face == false);

    return true;
}

bool test_plane_ray_interval_excludes() {
    Plane plane(Point3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), nullptr);

    // Ray would hit at t=5, but interval only allows [0, 3]
    Ray ray(Point3(0.0f, 5.0f, 0.0f), Vec3(0.0f, -1.0f, 0.0f));
    Interval ray_t(0.001f, 3.0f);
    HitRecord rec;

    bool hit = plane.hit(ray, ray_t, rec);
    TEST_ASSERT(hit == false);

    return true;
}

// ==============================================================================
// Normal Tests
// ==============================================================================

bool test_plane_normal_front_face() {
    Plane plane(Point3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), nullptr);

    // Ray from above (front face)
    Ray ray(Point3(0.0f, 5.0f, 0.0f), Vec3(0.0f, -1.0f, 0.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    plane.hit(ray, ray_t, rec);

    // Normal should point up (toward ray)
    TEST_ASSERT_NEAR(rec.normal.y, 1.0f, EPS);
    TEST_ASSERT(rec.front_face == true);

    return true;
}

bool test_plane_normal_back_face() {
    Plane plane(Point3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), nullptr);

    // Ray from below (back face)
    Ray ray(Point3(0.0f, -5.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    plane.hit(ray, ray_t, rec);

    // Normal should be flipped to point toward ray (down)
    TEST_ASSERT_NEAR(rec.normal.y, -1.0f, EPS);
    TEST_ASSERT(rec.front_face == false);

    return true;
}

bool test_plane_normal_is_unit() {
    // Plane with non-unit normal (should be normalized in constructor)
    Plane plane(Point3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 5.0f, 0.0f), nullptr);

    Ray ray(Point3(0.0f, 5.0f, 0.0f), Vec3(0.0f, -1.0f, 0.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    plane.hit(ray, ray_t, rec);

    float len = rec.normal.length();
    TEST_ASSERT_NEAR(len, 1.0f, EPS);

    return true;
}

// ==============================================================================
// Different Plane Orientations
// ==============================================================================

bool test_plane_vertical_xz() {
    // Vertical plane at x=3, facing +x
    Plane plane(Point3(3.0f, 0.0f, 0.0f), Vec3(1.0f, 0.0f, 0.0f), nullptr);

    Ray ray(Point3(0.0f, 0.0f, 0.0f), Vec3(1.0f, 0.0f, 0.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = plane.hit(ray, ray_t, rec);
    TEST_ASSERT(hit == true);
    TEST_ASSERT_NEAR(rec.t, 3.0f, EPS);
    TEST_ASSERT_NEAR(rec.p.x, 3.0f, EPS);

    return true;
}

bool test_plane_diagonal() {
    // Diagonal plane with normal (1, 1, 0) normalized
    Vec3 normal = Vec3(1.0f, 1.0f, 0.0f).normalized();
    Plane plane(Point3(5.0f, 5.0f, 0.0f), normal, nullptr);

    // Ray from origin toward plane
    Ray ray(Point3(0.0f, 0.0f, 0.0f), Vec3(1.0f, 1.0f, 0.0f).normalized());
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = plane.hit(ray, ray_t, rec);
    TEST_ASSERT(hit == true);

    // Hit point should be on the plane: (p - point) · normal = 0
    Vec3 diff = rec.p - Point3(5.0f, 5.0f, 0.0f);
    float check = dot(diff, normal);
    TEST_ASSERT_NEAR(check, 0.0f, EPS);

    return true;
}

bool test_plane_offset_point() {
    // Horizontal plane at y=10
    Plane plane(Point3(0.0f, 10.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), nullptr);

    Ray ray(Point3(0.0f, 15.0f, 0.0f), Vec3(0.0f, -1.0f, 0.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = plane.hit(ray, ray_t, rec);
    TEST_ASSERT(hit == true);
    TEST_ASSERT_NEAR(rec.p.y, 10.0f, EPS);
    TEST_ASSERT_NEAR(rec.t, 5.0f, EPS);

    return true;
}

// ==============================================================================
// Test Suite Runner
// ==============================================================================

void run_plane_tests(int& passed, int& failed, int& total) {
    // Ray-Plane Intersection
    RUN_TEST(test_plane_ray_perpendicular);
    RUN_TEST(test_plane_ray_oblique);
    RUN_TEST(test_plane_ray_parallel);
    RUN_TEST(test_plane_ray_pointing_away);
    RUN_TEST(test_plane_ray_from_behind);
    RUN_TEST(test_plane_ray_interval_excludes);

    // Normals
    RUN_TEST(test_plane_normal_front_face);
    RUN_TEST(test_plane_normal_back_face);
    RUN_TEST(test_plane_normal_is_unit);

    // Different orientations
    RUN_TEST(test_plane_vertical_xz);
    RUN_TEST(test_plane_diagonal);
    RUN_TEST(test_plane_offset_point);
}
