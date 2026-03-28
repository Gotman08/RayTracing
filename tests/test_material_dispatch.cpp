/**
 * @file test_material_dispatch.cpp
 * @brief Tests scatter_cpu() : dispatch lambertian/metal/dielectric
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

#include "raytracer/rendering/material_dispatch.cuh"

using namespace rt;


/** @brief Dispatch lambertian -> true, attenuation = albedo */
TEST(MaterialDispatchTest, LambertianDispatch) {
    Material mat(MaterialType::LAMBERTIAN, Color(0.8f, 0.2f, 0.1f));

    Ray r_in(Point3(0, 0, -1), Vec3(0, 0, 1));
    HitRecord rec;
    rec.p = Point3(0, 0, 0);
    rec.normal = Vec3(0, 0, -1);
    rec.t = 1.0f;
    rec.front_face = true;
    rec.mat = &mat;

    Color attenuation;
    Ray scattered(Point3(0, 0, 0), Vec3(0, 0, 0));
    CPURandom rng(42);

    bool result = scatter_cpu(mat, r_in, rec, attenuation, scattered, rng);
    EXPECT_TRUE(result);
    EXPECT_NEAR(attenuation.x, 0.8f, 1e-5f);
    EXPECT_NEAR(attenuation.y, 0.2f, 1e-5f);
    EXPECT_NEAR(attenuation.z, 0.1f, 1e-5f);
}

/** @brief Dispatch metal fuzz=0 a 45deg -> true, reflechi vers le haut */
TEST(MaterialDispatchTest, MetalDispatch) {
    Material mat(MaterialType::METAL, Color(0.9f, 0.9f, 0.9f), 0.0f);

    // Rayon a 45 degres : direction (-1, -1, 0) normalisee
    float inv_sqrt2 = 1.0f / sqrtf(2.0f);
    Ray r_in(Point3(-1, 1, 0), Vec3(inv_sqrt2, -inv_sqrt2, 0.0f));
    HitRecord rec;
    rec.p = Point3(0, 0, 0);
    rec.normal = Vec3(0, 1, 0);
    rec.t = 1.0f;
    rec.front_face = true;
    rec.mat = &mat;

    Color attenuation;
    Ray scattered(Point3(0, 0, 0), Vec3(0, 0, 0));
    CPURandom rng(42);

    bool result = scatter_cpu(mat, r_in, rec, attenuation, scattered, rng);
    EXPECT_TRUE(result);
    // Le rayon reflechi doit avoir une composante Y positive
    EXPECT_GT(scattered.direction().y, 0.0f);
}

/** @brief Dispatch dielectrique ior=1.5 -> true, attenuation blanche */
TEST(MaterialDispatchTest, DielectricDispatch) {
    Material mat = Material::make_dielectric(1.5f);

    Ray r_in(Point3(0, 0, -2), Vec3(0, 0, 1));
    HitRecord rec;
    rec.p = Point3(0, 0, 0);
    rec.normal = Vec3(0, 0, -1);
    rec.t = 2.0f;
    rec.front_face = true;
    rec.mat = &mat;

    Color attenuation;
    Ray scattered(Point3(0, 0, 0), Vec3(0, 0, 0));
    CPURandom rng(42);

    bool result = scatter_cpu(mat, r_in, rec, attenuation, scattered, rng);
    EXPECT_TRUE(result);
    // L'attenuation d'un dielectrique est toujours blanche
    EXPECT_NEAR(attenuation.x, 1.0f, 1e-5f);
    EXPECT_NEAR(attenuation.y, 1.0f, 1e-5f);
    EXPECT_NEAR(attenuation.z, 1.0f, 1e-5f);
}

/** @brief Attenuation via dispatch = mat.albedo exact */
TEST(MaterialDispatchTest, AttenuationMatchesAlbedo) {
    Color albedo(0.3f, 0.6f, 0.9f);
    Material mat(MaterialType::LAMBERTIAN, albedo);

    Ray r_in(Point3(0, 1, 0), Vec3(0, -1, 0));
    HitRecord rec;
    rec.p = Point3(0, 0, 0);
    rec.normal = Vec3(0, 1, 0);
    rec.t = 1.0f;
    rec.front_face = true;
    rec.mat = &mat;

    Color attenuation;
    Ray scattered(Point3(0, 0, 0), Vec3(0, 0, 0));
    CPURandom rng(42);

    scatter_cpu(mat, r_in, rec, attenuation, scattered, rng);
    EXPECT_NEAR(attenuation.x, mat.albedo.x, 1e-5f);
    EXPECT_NEAR(attenuation.y, mat.albedo.y, 1e-5f);
    EXPECT_NEAR(attenuation.z, mat.albedo.z, 1e-5f);
}
