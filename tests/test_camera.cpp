/**
 * Unit Tests - Camera
 */

#include <iostream>
#include <cmath>

#include "raytracer/core/vec3.cuh"
#include "raytracer/core/ray.cuh"
#include "raytracer/core/random.cuh"

// Stub CUDA random functions for host-only compilation
#ifndef __CUDACC__
inline float curand_uniform(curandState*) { return 0.5f; }
namespace rt {
    inline Vec3 random_unit_vector(curandState*) { return Vec3(0, 1, 0); }
    inline Vec3 random_in_unit_sphere(curandState*) { return Vec3(0, 0, 0); }
    inline Vec3 random_in_unit_disk(curandState*) { return Vec3(0, 0, 0); }
}
#endif

#include "raytracer/camera/camera.cuh"

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
// Camera initialization tests
// ============================================================================

static bool test_camera_center() {
    Camera cam;
    cam.initialize(800, 600, Point3(0, 0, 0), Point3(0, 0, -1), Vec3(0, 1, 0), 90.0f);
    TEST_ASSERT_NEAR(cam.center.x, 0.0f, 1e-5f);
    TEST_ASSERT_NEAR(cam.center.y, 0.0f, 1e-5f);
    TEST_ASSERT_NEAR(cam.center.z, 0.0f, 1e-5f);
    return true;
}

static bool test_camera_dimensions() {
    Camera cam;
    cam.initialize(800, 600, Point3(0, 0, 0), Point3(0, 0, -1), Vec3(0, 1, 0), 90.0f);
    TEST_ASSERT(cam.image_width == 800);
    TEST_ASSERT(cam.image_height == 600);
    return true;
}

static bool test_camera_coordinate_system() {
    Camera cam;
    cam.initialize(800, 600, Point3(0, 0, 0), Point3(0, 0, -1), Vec3(0, 1, 0), 90.0f);
    // w should point away from target (positive z)
    TEST_ASSERT_NEAR(cam.w.z, 1.0f, 1e-5f);
    // u should be right (positive x)
    TEST_ASSERT_NEAR(cam.u.x, 1.0f, 1e-5f);
    // v should be up (positive y)
    TEST_ASSERT_NEAR(cam.v.y, 1.0f, 1e-5f);
    return true;
}

static bool test_camera_no_defocus() {
    Camera cam;
    cam.initialize(800, 600, Point3(0, 0, 0), Point3(0, 0, -1), Vec3(0, 1, 0), 90.0f, 0.0f);
    TEST_ASSERT_NEAR(cam.defocus_angle, 0.0f, 1e-5f);
    return true;
}

static bool test_camera_with_defocus() {
    Camera cam;
    cam.initialize(800, 600, Point3(0, 0, 0), Point3(0, 0, -1), Vec3(0, 1, 0), 90.0f, 2.0f, 10.0f);
    TEST_ASSERT_NEAR(cam.defocus_angle, 2.0f, 1e-5f);
    // defocus_disk_u and defocus_disk_v should be non-zero
    TEST_ASSERT(cam.defocus_disk_u.length() > 0.0f);
    TEST_ASSERT(cam.defocus_disk_v.length() > 0.0f);
    return true;
}

// ============================================================================
// Ray generation tests
// ============================================================================

static bool test_camera_ray_from_center() {
    Camera cam;
    cam.initialize(800, 600, Point3(0, 0, 0), Point3(0, 0, -1), Vec3(0, 1, 0), 90.0f);
    CPURandom rng(42);
    // Center pixel should generate a ray roughly along -z
    Ray r = cam.get_ray_cpu(400, 300, rng);
    TEST_ASSERT_NEAR(r.origin().x, 0.0f, 1e-5f);
    TEST_ASSERT_NEAR(r.origin().y, 0.0f, 1e-5f);
    TEST_ASSERT_NEAR(r.origin().z, 0.0f, 1e-5f);
    // Direction should be approximately (0, 0, -1) for center pixel
    Vec3 dir = unit_vector(r.direction());
    TEST_ASSERT(dir.z < -0.5f);
    return true;
}

static bool test_camera_ray_corner_diverges() {
    Camera cam;
    cam.initialize(800, 600, Point3(0, 0, 0), Point3(0, 0, -1), Vec3(0, 1, 0), 90.0f);
    CPURandom rng(42);
    // Top-left pixel should have a different direction than bottom-right
    Ray r1 = cam.get_ray_cpu(0, 0, rng);
    Ray r2 = cam.get_ray_cpu(799, 599, rng);
    Vec3 d1 = unit_vector(r1.direction());
    Vec3 d2 = unit_vector(r2.direction());
    // They should diverge significantly
    float cosine = dot(d1, d2);
    TEST_ASSERT(cosine < 0.9f);
    return true;
}

static bool test_camera_shutter_time() {
    Camera cam;
    cam.initialize(800, 600, Point3(0, 0, 0), Point3(0, 0, -1), Vec3(0, 1, 0), 90.0f,
                   0.0f, 10.0f, 0.0f, 1.0f);
    CPURandom rng(42);
    Ray r = cam.get_ray_cpu(400, 300, rng);
    TEST_ASSERT(r.time() >= 0.0f);
    TEST_ASSERT(r.time() <= 1.0f);
    return true;
}

static bool test_camera_fov_affects_spread() {
    Camera cam_narrow, cam_wide;
    cam_narrow.initialize(800, 600, Point3(0, 0, 0), Point3(0, 0, -1), Vec3(0, 1, 0), 20.0f);
    cam_wide.initialize(800, 600, Point3(0, 0, 0), Point3(0, 0, -1), Vec3(0, 1, 0), 90.0f);
    // Pixel deltas should be larger for wider FOV
    float narrow_delta = cam_narrow.pixel_delta_u.length();
    float wide_delta = cam_wide.pixel_delta_u.length();
    TEST_ASSERT(wide_delta > narrow_delta);
    return true;
}

// ============================================================================
// Test runner
// ============================================================================

void run_camera_tests(int& passed, int& failed, int& total) {
    RUN_TEST(test_camera_center);
    RUN_TEST(test_camera_dimensions);
    RUN_TEST(test_camera_coordinate_system);
    RUN_TEST(test_camera_no_defocus);
    RUN_TEST(test_camera_with_defocus);
    RUN_TEST(test_camera_ray_from_center);
    RUN_TEST(test_camera_ray_corner_diverges);
    RUN_TEST(test_camera_shutter_time);
    RUN_TEST(test_camera_fov_affects_spread);
}
