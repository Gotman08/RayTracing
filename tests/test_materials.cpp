/**
 * @file test_materials.cpp
 * @brief Tests materiaux : lambertian, metal, dielectric, scatter, Schlick
 */

#include <gtest/gtest.h>
#include <cmath>

#include "raytracer/core/vec3.cuh"
#include "raytracer/core/ray.cuh"
#include "raytracer/core/hit_record.cuh"
#include "raytracer/core/random.cuh"

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


/** @brief Cree lambertian -> type + albedo OK */
TEST(MaterialsTest, CreateLambertian) {
    Material mat = create_lambertian(Color(0.5f, 0.3f, 0.1f));
    EXPECT_TRUE(mat.type == MaterialType::LAMBERTIAN);
    EXPECT_NEAR(mat.albedo.x, 0.5f, 1e-5f);
    EXPECT_NEAR(mat.albedo.y, 0.3f, 1e-5f);
    EXPECT_NEAR(mat.albedo.z, 0.1f, 1e-5f);
}

/** @brief Cree metal -> albedo + fuzz stockes */
TEST(MaterialsTest, CreateMetal) {
    Material mat = create_metal(Color(0.8f, 0.8f, 0.8f), 0.3f);
    EXPECT_TRUE(mat.type == MaterialType::METAL);
    EXPECT_NEAR(mat.albedo.x, 0.8f, 1e-5f);
    EXPECT_NEAR(mat.fuzz, 0.3f, 1e-5f);
}

/** @brief Metal avec fuzz>1 -> clamp a 1.0 */
TEST(MaterialsTest, CreateMetalFuzzClamp) {
    Material mat = create_metal(Color(0.8f, 0.8f, 0.8f), 5.0f);
    EXPECT_TRUE(mat.fuzz <= 1.0f);
}

/** @brief Dielectrique ior=1.5 -> albedo blanc, type OK */
TEST(MaterialsTest, CreateDielectric) {
    Material mat = create_dielectric(1.5f);
    EXPECT_TRUE(mat.type == MaterialType::DIELECTRIC);
    EXPECT_NEAR(mat.ior, 1.5f, 1e-5f);
    EXPECT_NEAR(mat.albedo.x, 1.0f, 1e-5f);
    EXPECT_NEAR(mat.albedo.y, 1.0f, 1e-5f);
    EXPECT_NEAR(mat.albedo.z, 1.0f, 1e-5f);
}

/** @brief Material() par defaut : lambertian, fuzz=0, ior=1.5 */
TEST(MaterialsTest, MaterialDefault) {
    Material mat;
    EXPECT_TRUE(mat.type == MaterialType::LAMBERTIAN);
    EXPECT_NEAR(mat.fuzz, 0.0f, 1e-5f);
    EXPECT_NEAR(mat.ior, 1.5f, 1e-5f);
}

/** @brief make_dielectric(2.4) -> type DIELECTRIC, ior=2.4 */
TEST(MaterialsTest, MakeDielectricStatic) {
    Material mat = Material::make_dielectric(2.4f);
    EXPECT_TRUE(mat.type == MaterialType::DIELECTRIC);
    EXPECT_NEAR(mat.ior, 2.4f, 1e-5f);
}


/** @brief Scatter lambertian -> toujours true */
TEST(MaterialsTest, LambertianScatterAlwaysTrue) {
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
    EXPECT_TRUE(result == true);
}

/** @brief Attenuation lambertian = albedo du mat */
TEST(MaterialsTest, LambertianAttenuationIsAlbedo) {
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
    EXPECT_NEAR(attenuation.x, 0.7f, 1e-5f);
    EXPECT_NEAR(attenuation.y, 0.3f, 1e-5f);
    EXPECT_NEAR(attenuation.z, 0.1f, 1e-5f);
}

/** @brief Rayon diffuse part du hit point exact */
TEST(MaterialsTest, LambertianScatterOriginAtHit) {
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
    EXPECT_NEAR(scattered.origin().x, 1.0f, 1e-5f);
    EXPECT_NEAR(scattered.origin().y, 2.0f, 1e-5f);
    EXPECT_NEAR(scattered.origin().z, 3.0f, 1e-5f);
}


/** @brief Metal fuzz=0 : reflexion parfaite, rayon vers le haut */
TEST(MaterialsTest, MetalScatterReflectionDirection) {
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
    EXPECT_TRUE(result == true);
    EXPECT_TRUE(scattered.direction().y > 0);
}

/** @brief Attenuation metal = son albedo */
TEST(MaterialsTest, MetalAttenuationIsAlbedo) {
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
    EXPECT_NEAR(attenuation.x, 0.8f, 1e-5f);
    EXPECT_NEAR(attenuation.y, 0.6f, 1e-5f);
    EXPECT_NEAR(attenuation.z, 0.2f, 1e-5f);
}


/** @brief Schlick cos=1, ior=1.5 -> R0 ~ 0.04 */
TEST(MaterialsTest, SchlickReflectanceAtZero) {
    float r = reflectance(1.0f, 1.5f);
    float r0 = (1.0f - 1.5f) / (1.0f + 1.5f);
    r0 = r0 * r0;
    EXPECT_NEAR(r, r0, 1e-5f);
}

/** @brief Schlick incidence rasante cos~0 -> reflectance > 0.9 */
TEST(MaterialsTest, SchlickReflectanceAtGrazing) {
    float r = reflectance(0.01f, 1.5f);
    EXPECT_TRUE(r > 0.9f);
}

/** @brief Scatter dielectrique -> toujours true (jamais absorbe) */
TEST(MaterialsTest, DielectricScatterAlwaysTrue) {
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
    EXPECT_TRUE(result == true);
}

/** @brief Attenuation dielectrique = blanc (1,1,1) */
TEST(MaterialsTest, DielectricAttenuationIsWhite) {
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
    EXPECT_NEAR(attenuation.x, 1.0f, 1e-5f);
    EXPECT_NEAR(attenuation.y, 1.0f, 1e-5f);
    EXPECT_NEAR(attenuation.z, 1.0f, 1e-5f);
}
