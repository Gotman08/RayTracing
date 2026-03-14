/**
 * Unit Tests - Materials (Lambertian, Metal, Dielectric)
 */

#include <iostream>
#include <cmath>

#include "raytracer/core/vec3.cuh"
#include "raytracer/core/ray.cuh"
#include "raytracer/core/hit_record.cuh"
#include "raytracer/core/random.cuh"

// Stub CUDA random functions for host-only compilation
// These are needed because __device__ is empty in host mode,
// so device scatter functions are compiled but CUDA random functions are guarded by __CUDACC__
#ifndef __CUDACC__
inline float curand_uniform(curandState*) { return 0.5f; }
namespace rt {
    inline Vec3 random_unit_vector(curandState*) { return Vec3(0, 1, 0); }
    inline Vec3 random_in_unit_sphere(curandState*) { return Vec3(0, 0, 0); }
    inline Vec3 random_in_unit_disk(curandState*) { return Vec3(0, 0, 0); }
}
#endif

#include "raytracer/materials/material.cuh"
#include "raytracer/materials/lambertian.cuh"
#include "raytracer/materials/metal.cuh"
#include "raytracer/materials/dielectric.cuh"

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
// Material creation tests
// ============================================================================

static bool test_create_lambertian() {
    Material mat = create_lambertian(Color(0.5f, 0.3f, 0.1f));
    TEST_ASSERT(mat.type == MaterialType::LAMBERTIAN);
    TEST_ASSERT_NEAR(mat.albedo.x, 0.5f, 1e-5f);
    TEST_ASSERT_NEAR(mat.albedo.y, 0.3f, 1e-5f);
    TEST_ASSERT_NEAR(mat.albedo.z, 0.1f, 1e-5f);
    return true;
}

static bool test_create_metal() {
    Material mat = create_metal(Color(0.8f, 0.8f, 0.8f), 0.3f);
    TEST_ASSERT(mat.type == MaterialType::METAL);
    TEST_ASSERT_NEAR(mat.albedo.x, 0.8f, 1e-5f);
    TEST_ASSERT_NEAR(mat.fuzz, 0.3f, 1e-5f);
    return true;
}

static bool test_create_metal_fuzz_clamp() {
    Material mat = create_metal(Color(0.8f, 0.8f, 0.8f), 5.0f);
    TEST_ASSERT(mat.fuzz <= 1.0f);
    return true;
}

static bool test_create_dielectric() {
    Material mat = create_dielectric(1.5f);
    TEST_ASSERT(mat.type == MaterialType::DIELECTRIC);
    TEST_ASSERT_NEAR(mat.ior, 1.5f, 1e-5f);
    TEST_ASSERT_NEAR(mat.albedo.x, 1.0f, 1e-5f);
    TEST_ASSERT_NEAR(mat.albedo.y, 1.0f, 1e-5f);
    TEST_ASSERT_NEAR(mat.albedo.z, 1.0f, 1e-5f);
    return true;
}

static bool test_material_default() {
    Material mat;
    TEST_ASSERT(mat.type == MaterialType::LAMBERTIAN);
    TEST_ASSERT_NEAR(mat.fuzz, 0.0f, 1e-5f);
    TEST_ASSERT_NEAR(mat.ior, 1.5f, 1e-5f);
    return true;
}

static bool test_make_dielectric_static() {
    Material mat = Material::make_dielectric(2.4f);
    TEST_ASSERT(mat.type == MaterialType::DIELECTRIC);
    TEST_ASSERT_NEAR(mat.ior, 2.4f, 1e-5f);
    return true;
}

// ============================================================================
// Lambertian scatter tests
// ============================================================================

static bool test_lambertian_scatter_always_true() {
    Material mat = create_lambertian(Color(0.5f, 0.5f, 0.5f));
    Ray r_in(Point3(0, 0, -1), Vec3(0, 0, 1));
    HitRecord rec;
    rec.p = Point3(0, 0, 0);
    rec.normal = Vec3(0, 0, -1);
    rec.front_face = true;
    rec.mat = &mat;

    Color attenuation;
    Ray scattered(Point3(0,0,0), Vec3(0,0,0));
    CPURandom rng(42);

    bool result = scatter_lambertian_cpu(mat, r_in, rec, attenuation, scattered, rng);
    TEST_ASSERT(result == true);
    return true;
}

static bool test_lambertian_attenuation_is_albedo() {
    Color albedo(0.7f, 0.3f, 0.1f);
    Material mat = create_lambertian(albedo);
    Ray r_in(Point3(0, 0, -1), Vec3(0, 0, 1));
    HitRecord rec;
    rec.p = Point3(0, 0, 0);
    rec.normal = Vec3(0, 0, -1);
    rec.front_face = true;

    Color attenuation;
    Ray scattered(Point3(0,0,0), Vec3(0,0,0));
    CPURandom rng(42);

    scatter_lambertian_cpu(mat, r_in, rec, attenuation, scattered, rng);
    TEST_ASSERT_NEAR(attenuation.x, 0.7f, 1e-5f);
    TEST_ASSERT_NEAR(attenuation.y, 0.3f, 1e-5f);
    TEST_ASSERT_NEAR(attenuation.z, 0.1f, 1e-5f);
    return true;
}

static bool test_lambertian_scatter_origin_at_hit() {
    Material mat = create_lambertian(Color(0.5f, 0.5f, 0.5f));
    Ray r_in(Point3(0, 0, -1), Vec3(0, 0, 1));
    HitRecord rec;
    rec.p = Point3(1, 2, 3);
    rec.normal = Vec3(0, 1, 0);
    rec.front_face = true;

    Color attenuation;
    Ray scattered(Point3(0,0,0), Vec3(0,0,0));
    CPURandom rng(42);

    scatter_lambertian_cpu(mat, r_in, rec, attenuation, scattered, rng);
    TEST_ASSERT_NEAR(scattered.origin().x, 1.0f, 1e-5f);
    TEST_ASSERT_NEAR(scattered.origin().y, 2.0f, 1e-5f);
    TEST_ASSERT_NEAR(scattered.origin().z, 3.0f, 1e-5f);
    return true;
}

// ============================================================================
// Metal scatter tests
// ============================================================================

static bool test_metal_scatter_reflection_direction() {
    Material mat = create_metal(Color(0.9f, 0.9f, 0.9f), 0.0f);
    Ray r_in(Point3(0, 1, -1), Vec3(0, -1, 0));
    HitRecord rec;
    rec.p = Point3(0, 0, 0);
    rec.normal = Vec3(0, 1, 0);
    rec.front_face = true;

    Color attenuation;
    Ray scattered(Point3(0,0,0), Vec3(0,0,0));
    CPURandom rng(42);

    bool result = scatter_metal_cpu(mat, r_in, rec, attenuation, scattered, rng);
    TEST_ASSERT(result == true);
    // With fuzz=0, reflected ray should point upward (reflected off horizontal surface)
    TEST_ASSERT(scattered.direction().y > 0);
    return true;
}

static bool test_metal_attenuation_is_albedo() {
    Color albedo(0.8f, 0.6f, 0.2f);
    Material mat = create_metal(albedo, 0.0f);
    Ray r_in(Point3(0, 1, 0), Vec3(0, -1, 0));
    HitRecord rec;
    rec.p = Point3(0, 0, 0);
    rec.normal = Vec3(0, 1, 0);
    rec.front_face = true;

    Color attenuation;
    Ray scattered(Point3(0,0,0), Vec3(0,0,0));
    CPURandom rng(42);

    scatter_metal_cpu(mat, r_in, rec, attenuation, scattered, rng);
    TEST_ASSERT_NEAR(attenuation.x, 0.8f, 1e-5f);
    TEST_ASSERT_NEAR(attenuation.y, 0.6f, 1e-5f);
    TEST_ASSERT_NEAR(attenuation.z, 0.2f, 1e-5f);
    return true;
}

// ============================================================================
// Dielectric tests
// ============================================================================

static bool test_schlick_reflectance_at_zero() {
    // At normal incidence (cos=1), reflectance should be R0
    float r = reflectance(1.0f, 1.5f);
    float r0 = (1.0f - 1.5f) / (1.0f + 1.5f);
    r0 = r0 * r0;
    TEST_ASSERT_NEAR(r, r0, 1e-5f);
    return true;
}

static bool test_schlick_reflectance_at_grazing() {
    // At grazing angle (cos≈0), reflectance should approach 1
    float r = reflectance(0.01f, 1.5f);
    TEST_ASSERT(r > 0.9f);
    return true;
}

static bool test_dielectric_scatter_always_true() {
    Material mat = create_dielectric(1.5f);
    Ray r_in(Point3(0, 0, -2), Vec3(0, 0, 1));
    HitRecord rec;
    rec.p = Point3(0, 0, 0);
    rec.normal = Vec3(0, 0, -1);
    rec.front_face = true;
    rec.mat = &mat;

    Color attenuation;
    Ray scattered(Point3(0,0,0), Vec3(0,0,0));
    CPURandom rng(42);

    bool result = scatter_dielectric_cpu(mat, r_in, rec, attenuation, scattered, rng);
    TEST_ASSERT(result == true);
    return true;
}

static bool test_dielectric_attenuation_is_white() {
    Material mat = create_dielectric(1.5f);
    Ray r_in(Point3(0, 0, -2), Vec3(0, 0, 1));
    HitRecord rec;
    rec.p = Point3(0, 0, 0);
    rec.normal = Vec3(0, 0, -1);
    rec.front_face = true;

    Color attenuation;
    Ray scattered(Point3(0,0,0), Vec3(0,0,0));
    CPURandom rng(42);

    scatter_dielectric_cpu(mat, r_in, rec, attenuation, scattered, rng);
    TEST_ASSERT_NEAR(attenuation.x, 1.0f, 1e-5f);
    TEST_ASSERT_NEAR(attenuation.y, 1.0f, 1e-5f);
    TEST_ASSERT_NEAR(attenuation.z, 1.0f, 1e-5f);
    return true;
}

// ============================================================================
// Test runner
// ============================================================================

void run_materials_tests(int& passed, int& failed, int& total) {
    RUN_TEST(test_create_lambertian);
    RUN_TEST(test_create_metal);
    RUN_TEST(test_create_metal_fuzz_clamp);
    RUN_TEST(test_create_dielectric);
    RUN_TEST(test_material_default);
    RUN_TEST(test_make_dielectric_static);
    RUN_TEST(test_lambertian_scatter_always_true);
    RUN_TEST(test_lambertian_attenuation_is_albedo);
    RUN_TEST(test_lambertian_scatter_origin_at_hit);
    RUN_TEST(test_metal_scatter_reflection_direction);
    RUN_TEST(test_metal_attenuation_is_albedo);
    RUN_TEST(test_schlick_reflectance_at_zero);
    RUN_TEST(test_schlick_reflectance_at_grazing);
    RUN_TEST(test_dielectric_scatter_always_true);
    RUN_TEST(test_dielectric_attenuation_is_white);
}
