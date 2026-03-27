/**
 * @file test_aabb.cpp
 * @brief Tests AABB : ctors, hit rayon, centroide, surface area, longest axis
 */

#include <gtest/gtest.h>
#include <cmath>

#include "raytracer/core/aabb.cuh"
#include "raytracer/core/cuda_utils.cuh"

using namespace rt;

constexpr float EPS = 1e-4f;


/** @brief AABB depuis 2 points -> min/max corrects */
TEST(AABBTest, PointConstructor) {
    Point3 a(0.0f, 0.0f, 0.0f);
    Point3 b(2.0f, 3.0f, 4.0f);
    AABB box(a, b);

    EXPECT_NEAR(box.x.min, 0.0f, EPS);
    EXPECT_NEAR(box.x.max, 2.0f, EPS);
    EXPECT_NEAR(box.y.min, 0.0f, EPS);
    EXPECT_NEAR(box.y.max, 3.0f, EPS);
    EXPECT_NEAR(box.z.min, 0.0f, EPS);
    EXPECT_NEAR(box.z.max, 4.0f, EPS);
}

/** @brief Points inverses -> bornes triees automatiquement */
TEST(AABBTest, PointConstructorReversed) {
    Point3 a(5.0f, 5.0f, 5.0f);
    Point3 b(0.0f, 0.0f, 0.0f);
    AABB box(a, b);

    EXPECT_NEAR(box.x.min, 0.0f, EPS);
    EXPECT_NEAR(box.x.max, 5.0f, EPS);
}

/** @brief Ctor 3 intervalles -> X/Y/Z OK */
TEST(AABBTest, IntervalConstructor) {
    Interval ix(0.0f, 2.0f);
    Interval iy(0.0f, 3.0f);
    Interval iz(0.0f, 4.0f);
    AABB box(ix, iy, iz);

    EXPECT_NEAR(box.x.min, 0.0f, EPS);
    EXPECT_NEAR(box.x.max, 2.0f, EPS);
    EXPECT_NEAR(box.y.min, 0.0f, EPS);
    EXPECT_NEAR(box.y.max, 3.0f, EPS);
    EXPECT_NEAR(box.z.min, 0.0f, EPS);
    EXPECT_NEAR(box.z.max, 4.0f, EPS);
}

/** @brief Fusion 2 AABB disjointes -> englobante correcte */
TEST(AABBTest, MergeConstructor) {
    AABB box1(Point3(0.0f, 0.0f, 0.0f), Point3(1.0f, 1.0f, 1.0f));
    AABB box2(Point3(2.0f, 2.0f, 2.0f), Point3(3.0f, 3.0f, 3.0f));
    AABB merged(box1, box2);

    EXPECT_NEAR(merged.x.min, 0.0f, EPS);
    EXPECT_NEAR(merged.x.max, 3.0f, EPS);
    EXPECT_NEAR(merged.y.min, 0.0f, EPS);
    EXPECT_NEAR(merged.y.max, 3.0f, EPS);
}


/** @brief Rayon axe Z plein centre -> hit */
TEST(AABBTest, HitThroughCenter) {
    AABB box(Point3(0.0f, 0.0f, 0.0f), Point3(2.0f, 2.0f, 2.0f));

    Ray ray(Point3(1.0f, 1.0f, -5.0f), Vec3(0.0f, 0.0f, 1.0f));
    Interval ray_t(0.001f, INFINITY_F);

    bool hit = box.hit(ray, ray_t);
    EXPECT_TRUE(hit == true);
}

/** @brief Rayon rasant le coin -> quand meme hit */
TEST(AABBTest, HitEdge) {
    AABB box(Point3(0.0f, 0.0f, 0.0f), Point3(2.0f, 2.0f, 2.0f));

    Ray ray(Point3(0.001f, 0.001f, -5.0f), Vec3(0.0f, 0.0f, 1.0f));
    Interval ray_t(0.001f, INFINITY_F);

    bool hit = box.hit(ray, ray_t);
    EXPECT_TRUE(hit == true);
}

/** @brief Rayon loin de la boite -> miss */
TEST(AABBTest, Miss) {
    AABB box(Point3(0.0f, 0.0f, 0.0f), Point3(2.0f, 2.0f, 2.0f));

    Ray ray(Point3(5.0f, 5.0f, -5.0f), Vec3(0.0f, 0.0f, 1.0f));
    Interval ray_t(0.001f, INFINITY_F);

    bool hit = box.hit(ray, ray_t);
    EXPECT_TRUE(hit == false);
}

/** @brief Rayon direction opposee -> miss */
TEST(AABBTest, RayBehind) {
    AABB box(Point3(0.0f, 0.0f, 0.0f), Point3(2.0f, 2.0f, 2.0f));

    Ray ray(Point3(1.0f, 1.0f, -5.0f), Vec3(0.0f, 0.0f, -1.0f));
    Interval ray_t(0.001f, INFINITY_F);

    bool hit = box.hit(ray, ray_t);
    EXPECT_TRUE(hit == false);
}

/** @brief Origine dans la boite -> hit en sortie */
TEST(AABBTest, RayInside) {
    AABB box(Point3(0.0f, 0.0f, 0.0f), Point3(10.0f, 10.0f, 10.0f));

    Ray ray(Point3(5.0f, 5.0f, 5.0f), Vec3(1.0f, 0.0f, 0.0f));
    Interval ray_t(0.001f, INFINITY_F);

    bool hit = box.hit(ray, ray_t);
    EXPECT_TRUE(hit == true);
}

/** @brief Rayon diagonal (-1,-1,-1)->(1,1,1) traverse la boite */
TEST(AABBTest, DiagonalRay) {
    AABB box(Point3(0.0f, 0.0f, 0.0f), Point3(2.0f, 2.0f, 2.0f));

    Ray ray(Point3(-1.0f, -1.0f, -1.0f), Vec3(1.0f, 1.0f, 1.0f).normalized());
    Interval ray_t(0.001f, INFINITY_F);

    bool hit = box.hit(ray, ray_t);
    EXPECT_TRUE(hit == true);
}

/** @brief Intervalle t trop court -> miss malgre geometrie OK */
TEST(AABBTest, IntervalExcludes) {
    AABB box(Point3(0.0f, 0.0f, 0.0f), Point3(2.0f, 2.0f, 2.0f));

    Ray ray(Point3(1.0f, 1.0f, -10.0f), Vec3(0.0f, 0.0f, 1.0f));
    Interval ray_t(0.001f, 5.0f);

    bool hit = box.hit(ray, ray_t);
    EXPECT_TRUE(hit == false);
}


/** @brief Centroide [0,4]x[0,6]x[0,8] = (2,3,4) */
TEST(AABBTest, Centroid) {
    AABB box(Point3(0.0f, 0.0f, 0.0f), Point3(4.0f, 6.0f, 8.0f));
    Point3 c = box.centroid();

    EXPECT_NEAR(c.x, 2.0f, EPS);
    EXPECT_NEAR(c.y, 3.0f, EPS);
    EXPECT_NEAR(c.z, 4.0f, EPS);
}

/** @brief Centroide boite decalee [2,4]^3 = (3,3,3) */
TEST(AABBTest, CentroidOffset) {
    AABB box(Point3(2.0f, 2.0f, 2.0f), Point3(4.0f, 4.0f, 4.0f));
    Point3 c = box.centroid();

    EXPECT_NEAR(c.x, 3.0f, EPS);
    EXPECT_NEAR(c.y, 3.0f, EPS);
    EXPECT_NEAR(c.z, 3.0f, EPS);
}


/** @brief Cube cote 2 -> surface_area = 24 */
TEST(AABBTest, SurfaceAreaCube) {
    AABB box(Point3(0.0f, 0.0f, 0.0f), Point3(2.0f, 2.0f, 2.0f));
    float sa = box.surface_area();

    EXPECT_NEAR(sa, 24.0f, EPS);
}

/** @brief Parallelepipede 2x3x4 -> surface_area = 52 */
TEST(AABBTest, SurfaceAreaRect) {
    AABB box(Point3(0.0f, 0.0f, 0.0f), Point3(2.0f, 3.0f, 4.0f));
    float sa = box.surface_area();

    EXPECT_NEAR(sa, 52.0f, EPS);
}


/** @brief Boite 10x2x3 -> longest_axis = 0 (X) */
TEST(AABBTest, LongestAxisX) {
    AABB box(Point3(0.0f, 0.0f, 0.0f), Point3(10.0f, 2.0f, 3.0f));
    EXPECT_TRUE(box.longest_axis() == 0);
}

/** @brief Boite 2x10x3 -> longest_axis = 1 (Y) */
TEST(AABBTest, LongestAxisY) {
    AABB box(Point3(0.0f, 0.0f, 0.0f), Point3(2.0f, 10.0f, 3.0f));
    EXPECT_TRUE(box.longest_axis() == 1);
}

/** @brief Boite 2x3x10 -> longest_axis = 2 (Z) */
TEST(AABBTest, LongestAxisZ) {
    AABB box(Point3(0.0f, 0.0f, 0.0f), Point3(2.0f, 3.0f, 10.0f));
    EXPECT_TRUE(box.longest_axis() == 2);
}


/** @brief axis_interval(0/1/2) renvoie X/Y/Z respectivement */
TEST(AABBTest, AxisInterval) {
    AABB box(Point3(1.0f, 2.0f, 3.0f), Point3(4.0f, 5.0f, 6.0f));

    const Interval& ix = box.axis_interval(0);
    const Interval& iy = box.axis_interval(1);
    const Interval& iz = box.axis_interval(2);

    EXPECT_NEAR(ix.min, 1.0f, EPS);
    EXPECT_NEAR(ix.max, 4.0f, EPS);
    EXPECT_NEAR(iy.min, 2.0f, EPS);
    EXPECT_NEAR(iy.max, 5.0f, EPS);
    EXPECT_NEAR(iz.min, 3.0f, EPS);
    EXPECT_NEAR(iz.max, 6.0f, EPS);
}
