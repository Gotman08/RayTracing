/**
 * @file test_ray.cpp
 * @brief Tests Ray : ctors, accesseurs, at(t)
 */

#include <gtest/gtest.h>
#include <cmath>

#include "raytracer/core/ray.cuh"

using namespace rt;

constexpr float EPS = 1e-6f;


/** @brief Ctor par defaut -> origin=0, dir=0, time=0 */
TEST(RayTest, DefaultConstructor) {
    Ray r;
    EXPECT_NEAR(r.origin().x, 0.0f, EPS);
    EXPECT_NEAR(r.origin().y, 0.0f, EPS);
    EXPECT_NEAR(r.origin().z, 0.0f, EPS);
    EXPECT_NEAR(r.direction().x, 0.0f, EPS);
    EXPECT_NEAR(r.direction().y, 0.0f, EPS);
    EXPECT_NEAR(r.direction().z, 0.0f, EPS);
    EXPECT_NEAR(r.time(), 0.0f, EPS);
}

/** @brief Ctor(origin, dir) stocke les valeurs, time=0 */
TEST(RayTest, ParameterizedConstructor) {
    Point3 origin(1.0f, 2.0f, 3.0f);
    Vec3 direction(0.0f, 0.0f, -1.0f);
    Ray r(origin, direction);

    EXPECT_NEAR(r.origin().x, 1.0f, EPS);
    EXPECT_NEAR(r.origin().y, 2.0f, EPS);
    EXPECT_NEAR(r.origin().z, 3.0f, EPS);
    EXPECT_NEAR(r.direction().x, 0.0f, EPS);
    EXPECT_NEAR(r.direction().y, 0.0f, EPS);
    EXPECT_NEAR(r.direction().z, -1.0f, EPS);
    EXPECT_NEAR(r.time(), 0.0f, EPS);
}

/** @brief Ctor avec time=0.5 pour motion blur */
TEST(RayTest, WithTime) {
    Point3 origin(0.0f, 0.0f, 0.0f);
    Vec3 direction(1.0f, 0.0f, 0.0f);
    Ray r(origin, direction, 0.5f);

    EXPECT_NEAR(r.time(), 0.5f, EPS);
}


/** @brief at(0) = origin */
TEST(RayTest, AtZero) {
    Point3 origin(1.0f, 2.0f, 3.0f);
    Vec3 direction(1.0f, 0.0f, 0.0f);
    Ray r(origin, direction);

    Point3 p = r.at(0.0f);
    EXPECT_NEAR(p.x, 1.0f, EPS);
    EXPECT_NEAR(p.y, 2.0f, EPS);
    EXPECT_NEAR(p.z, 3.0f, EPS);
}

/** @brief at(2) = origin + 2*dir */
TEST(RayTest, AtPositive) {
    Point3 origin(0.0f, 0.0f, 0.0f);
    Vec3 direction(1.0f, 2.0f, 3.0f);
    Ray r(origin, direction);

    Point3 p = r.at(2.0f);
    EXPECT_NEAR(p.x, 2.0f, EPS);
    EXPECT_NEAR(p.y, 4.0f, EPS);
    EXPECT_NEAR(p.z, 6.0f, EPS);
}

/** @brief at(-3) -> point derriere l'origine */
TEST(RayTest, AtNegative) {
    Point3 origin(5.0f, 5.0f, 5.0f);
    Vec3 direction(1.0f, 0.0f, 0.0f);
    Ray r(origin, direction);

    Point3 p = r.at(-3.0f);
    EXPECT_NEAR(p.x, 2.0f, EPS);
    EXPECT_NEAR(p.y, 5.0f, EPS);
    EXPECT_NEAR(p.z, 5.0f, EPS);
}

/** @brief at(0.5) -> mi-chemin le long du rayon */
TEST(RayTest, AtFractional) {
    Point3 origin(0.0f, 0.0f, 0.0f);
    Vec3 direction(4.0f, 0.0f, 0.0f);
    Ray r(origin, direction);

    Point3 p = r.at(0.5f);
    EXPECT_NEAR(p.x, 2.0f, EPS);
    EXPECT_NEAR(p.y, 0.0f, EPS);
    EXPECT_NEAR(p.z, 0.0f, EPS);
}

/** @brief dir unitaire + at(1) -> dist=1 de l'origin */
TEST(RayTest, AtUnitDirection) {
    Point3 origin(0.0f, 0.0f, 0.0f);
    Vec3 direction = Vec3(1.0f, 1.0f, 1.0f).normalized();
    Ray r(origin, direction);

    Point3 p = r.at(1.0f);
    float dist = (p - origin).length();
    EXPECT_NEAR(dist, 1.0f, EPS);
}


/** @brief rayon diagonal (1,1,1) : at(3) avance sur les 3 axes */
TEST(RayTest, DiagonalDirection) {
    Point3 origin(1.0f, 1.0f, 1.0f);
    Vec3 direction(1.0f, 1.0f, 1.0f);
    Ray r(origin, direction);

    Point3 p = r.at(3.0f);
    EXPECT_NEAR(p.x, 4.0f, EPS);
    EXPECT_NEAR(p.y, 4.0f, EPS);
    EXPECT_NEAR(p.z, 4.0f, EPS);
}

/** @brief rayon -Z typique camera, at(10) -> z=-10 */
TEST(RayTest, ZDirection) {
    Point3 origin(0.0f, 0.0f, 0.0f);
    Vec3 direction(0.0f, 0.0f, -1.0f);
    Ray r(origin, direction);

    Point3 p = r.at(10.0f);
    EXPECT_NEAR(p.x, 0.0f, EPS);
    EXPECT_NEAR(p.y, 0.0f, EPS);
    EXPECT_NEAR(p.z, -10.0f, EPS);
}
