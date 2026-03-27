/**
 * @file test_sphere.cpp
 * @brief Tests Sphere::hit() : intersections, normales, cas limites
 */

#include <gtest/gtest.h>
#include <cmath>

#include "raytracer/core/ray.cuh"
#include "raytracer/core/interval.cuh"
#include "raytracer/core/hit_record.cuh"
#include "raytracer/geometry/sphere.cuh"

using namespace rt;

constexpr float EPS = 1e-5f;


/** @brief Rayon plein centre -> t=4, impact z=1 */
TEST(SphereTest, RayThroughCenter) {
    Sphere sphere(Point3(0.0f, 0.0f, 0.0f), 1.0f, nullptr);

    Ray ray(Point3(0.0f, 0.0f, 5.0f), Vec3(0.0f, 0.0f, -1.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = sphere.hit(ray, ray_t, rec);
    EXPECT_TRUE(hit == true);

    EXPECT_NEAR(rec.t, 4.0f, EPS);
    EXPECT_NEAR(rec.p.z, 1.0f, EPS);
}

/** @brief Rayon decale x=2 -> miss */
TEST(SphereTest, RayMisses) {
    Sphere sphere(Point3(0.0f, 0.0f, 0.0f), 1.0f, nullptr);

    Ray ray(Point3(2.0f, 0.0f, 5.0f), Vec3(0.0f, 0.0f, -1.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = sphere.hit(ray, ray_t, rec);
    EXPECT_TRUE(hit == false);
}

/** @brief Rayon tangent a la sphere -> t unique, discriminant=0 */
TEST(SphereTest, RayTangent) {
    Sphere sphere(Point3(0.0f, 0.0f, 0.0f), 1.0f, nullptr);

    Ray ray(Point3(1.0f, 0.0f, 5.0f), Vec3(0.0f, 0.0f, -1.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = sphere.hit(ray, ray_t, rec);
    EXPECT_TRUE(hit == true);

    EXPECT_NEAR(rec.p.x, 1.0f, EPS);
    EXPECT_NEAR(rec.p.z, 0.0f, EPS);
}

/** @brief Rayon depuis l'interieur -> racine positive, t=2 */
TEST(SphereTest, RayFromInside) {
    Sphere sphere(Point3(0.0f, 0.0f, 0.0f), 2.0f, nullptr);

    Ray ray(Point3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, 1.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = sphere.hit(ray, ray_t, rec);
    EXPECT_TRUE(hit == true);

    EXPECT_NEAR(rec.t, 2.0f, EPS);
    EXPECT_NEAR(rec.p.z, 2.0f, EPS);
}

/** @brief Sphere derriere le rayon -> miss */
TEST(SphereTest, RayBehind) {
    Sphere sphere(Point3(0.0f, 0.0f, 0.0f), 1.0f, nullptr);

    Ray ray(Point3(0.0f, 0.0f, 5.0f), Vec3(0.0f, 0.0f, 1.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = sphere.hit(ray, ray_t, rec);
    EXPECT_TRUE(hit == false);
}

/** @brief t=4 hors intervalle [0.001,3] -> rejete */
TEST(SphereTest, RayIntervalExcludes) {
    Sphere sphere(Point3(0.0f, 0.0f, 0.0f), 1.0f, nullptr);

    Ray ray(Point3(0.0f, 0.0f, 5.0f), Vec3(0.0f, 0.0f, -1.0f));
    Interval ray_t(0.001f, 3.0f);
    HitRecord rec;

    bool hit = sphere.hit(ray, ray_t, rec);
    EXPECT_TRUE(hit == false);
}


/** @brief Normale face avant -> n=(0,0,1), front_face=true */
TEST(SphereTest, NormalAtFront) {
    Sphere sphere(Point3(0.0f, 0.0f, 0.0f), 1.0f, nullptr);

    Ray ray(Point3(0.0f, 0.0f, 5.0f), Vec3(0.0f, 0.0f, -1.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    sphere.hit(ray, ray_t, rec);

    EXPECT_NEAR(rec.normal.x, 0.0f, EPS);
    EXPECT_NEAR(rec.normal.y, 0.0f, EPS);
    EXPECT_NEAR(rec.normal.z, 1.0f, EPS);
    EXPECT_TRUE(rec.front_face == true);
}

/** @brief Rayon interne -> normale inversee, front_face=false */
TEST(SphereTest, NormalFromInside) {
    Sphere sphere(Point3(0.0f, 0.0f, 0.0f), 2.0f, nullptr);

    Ray ray(Point3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, 1.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    sphere.hit(ray, ray_t, rec);

    EXPECT_NEAR(rec.normal.z, -1.0f, EPS);
    EXPECT_TRUE(rec.front_face == false);
}

/** @brief Normale toujours unitaire quel que soit le rayon */
TEST(SphereTest, NormalIsUnit) {
    Sphere sphere(Point3(1.0f, 2.0f, 3.0f), 2.5f, nullptr);

    Ray ray(Point3(10.0f, 10.0f, 10.0f), Vec3(-1.0f, -1.0f, -1.0f).normalized());
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = sphere.hit(ray, ray_t, rec);
    if (!hit) FAIL() << "Ray did not hit sphere";

    float len = rec.normal.length();
    EXPECT_NEAR(len, 1.0f, EPS);
}


/** @brief Sphere decalee en (5,0,0) -> hit OK */
TEST(SphereTest, OffsetCenter) {
    Sphere sphere(Point3(5.0f, 0.0f, 0.0f), 1.0f, nullptr);

    Ray ray(Point3(5.0f, 0.0f, 5.0f), Vec3(0.0f, 0.0f, -1.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = sphere.hit(ray, ray_t, rec);
    EXPECT_TRUE(hit == true);
    EXPECT_NEAR(rec.p.x, 5.0f, EPS);
    EXPECT_NEAR(rec.p.z, 1.0f, EPS);
}

/** @brief Grande sphere r=100 -> precision OK, t=100 */
TEST(SphereTest, LargeRadius) {
    Sphere sphere(Point3(0.0f, 0.0f, 0.0f), 100.0f, nullptr);

    Ray ray(Point3(0.0f, 0.0f, 200.0f), Vec3(0.0f, 0.0f, -1.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = sphere.hit(ray, ray_t, rec);
    EXPECT_TRUE(hit == true);
    EXPECT_NEAR(rec.t, 100.0f, EPS);
}

/** @brief Micro sphere r=0.01 -> precision OK */
TEST(SphereTest, SmallRadius) {
    Sphere sphere(Point3(0.0f, 0.0f, 0.0f), 0.01f, nullptr);

    Ray ray(Point3(0.0f, 0.0f, 1.0f), Vec3(0.0f, 0.0f, -1.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = sphere.hit(ray, ray_t, rec);
    EXPECT_TRUE(hit == true);
    EXPECT_NEAR(rec.p.z, 0.01f, EPS);
}
