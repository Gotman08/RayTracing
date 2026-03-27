/**
 * @file test_plane.cpp
 * @brief Tests Plane::hit() : perp, oblique, parallele, normales
 */

#include <gtest/gtest.h>
#include <cmath>

#include "raytracer/core/ray.cuh"
#include "raytracer/core/interval.cuh"
#include "raytracer/core/hit_record.cuh"
#include "raytracer/geometry/plane.cuh"

using namespace rt;

constexpr float EPS = 1e-5f;


/** @brief Rayon perp au sol y=0 -> t=5 depuis y=5 */
TEST(PlaneTest, RayPerpendicular) {
    Plane plane(Point3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), nullptr);

    Ray ray(Point3(0.0f, 5.0f, 0.0f), Vec3(0.0f, -1.0f, 0.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = plane.hit(ray, ray_t, rec);
    EXPECT_TRUE(hit == true);

    EXPECT_NEAR(rec.t, 5.0f, EPS);
    EXPECT_NEAR(rec.p.y, 0.0f, EPS);
}

/** @brief Rayon oblique (0,-1,-1) touche y=0 */
TEST(PlaneTest, RayOblique) {
    Plane plane(Point3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), nullptr);

    Ray ray(Point3(0.0f, 5.0f, 5.0f), Vec3(0.0f, -1.0f, -1.0f).normalized());
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = plane.hit(ray, ray_t, rec);
    EXPECT_TRUE(hit == true);

    EXPECT_NEAR(rec.p.y, 0.0f, EPS);
}

/** @brief Rayon parallele au plan -> miss */
TEST(PlaneTest, RayParallel) {
    Plane plane(Point3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), nullptr);

    Ray ray(Point3(0.0f, 5.0f, 0.0f), Vec3(1.0f, 0.0f, 0.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = plane.hit(ray, ray_t, rec);
    EXPECT_TRUE(hit == false);
}

/** @brief Rayon qui s'eloigne du plan -> miss */
TEST(PlaneTest, RayPointingAway) {
    Plane plane(Point3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), nullptr);

    Ray ray(Point3(0.0f, 5.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = plane.hit(ray, ray_t, rec);
    EXPECT_TRUE(hit == false);
}

/** @brief Rayon par l'arriere -> hit, front_face=false */
TEST(PlaneTest, RayFromBehind) {
    Plane plane(Point3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), nullptr);

    Ray ray(Point3(0.0f, -5.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = plane.hit(ray, ray_t, rec);
    EXPECT_TRUE(hit == true);

    EXPECT_TRUE(rec.front_face == false);
}

/** @brief t=5 hors intervalle [0.001,3] -> rejete */
TEST(PlaneTest, RayIntervalExcludes) {
    Plane plane(Point3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), nullptr);

    Ray ray(Point3(0.0f, 5.0f, 0.0f), Vec3(0.0f, -1.0f, 0.0f));
    Interval ray_t(0.001f, 3.0f);
    HitRecord rec;

    bool hit = plane.hit(ray, ray_t, rec);
    EXPECT_TRUE(hit == false);
}


/** @brief Face avant -> n.y=1, front_face=true */
TEST(PlaneTest, NormalFrontFace) {
    Plane plane(Point3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), nullptr);

    Ray ray(Point3(0.0f, 5.0f, 0.0f), Vec3(0.0f, -1.0f, 0.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    plane.hit(ray, ray_t, rec);

    EXPECT_NEAR(rec.normal.y, 1.0f, EPS);
    EXPECT_TRUE(rec.front_face == true);
}

/** @brief Face arriere -> n.y=-1 inversee, front_face=false */
TEST(PlaneTest, NormalBackFace) {
    Plane plane(Point3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), nullptr);

    Ray ray(Point3(0.0f, -5.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    plane.hit(ray, ray_t, rec);

    EXPECT_NEAR(rec.normal.y, -1.0f, EPS);
    EXPECT_TRUE(rec.front_face == false);
}

/** @brief Normale non-unitaire au ctor -> unitaire dans HitRecord */
TEST(PlaneTest, NormalIsUnit) {
    Plane plane(Point3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 5.0f, 0.0f), nullptr);

    Ray ray(Point3(0.0f, 5.0f, 0.0f), Vec3(0.0f, -1.0f, 0.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    plane.hit(ray, ray_t, rec);

    float len = rec.normal.length();
    EXPECT_NEAR(len, 1.0f, EPS);
}


/** @brief Plan vertical x=3 -> rayon horizontal touche a t=3 */
TEST(PlaneTest, VerticalXz) {
    Plane plane(Point3(3.0f, 0.0f, 0.0f), Vec3(1.0f, 0.0f, 0.0f), nullptr);

    Ray ray(Point3(0.0f, 0.0f, 0.0f), Vec3(1.0f, 0.0f, 0.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = plane.hit(ray, ray_t, rec);
    EXPECT_TRUE(hit == true);
    EXPECT_NEAR(rec.t, 3.0f, EPS);
    EXPECT_NEAR(rec.p.x, 3.0f, EPS);
}

/** @brief Plan diagonal 45deg -> dot(impact-ref, n) = 0 */
TEST(PlaneTest, Diagonal) {
    Vec3 normal = Vec3(1.0f, 1.0f, 0.0f).normalized();
    Plane plane(Point3(5.0f, 5.0f, 0.0f), normal, nullptr);

    Ray ray(Point3(0.0f, 0.0f, 0.0f), Vec3(1.0f, 1.0f, 0.0f).normalized());
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = plane.hit(ray, ray_t, rec);
    EXPECT_TRUE(hit == true);

    Vec3 diff = rec.p - Point3(5.0f, 5.0f, 0.0f);
    float check = dot(diff, normal);
    EXPECT_NEAR(check, 0.0f, EPS);
}

/** @brief Plan decale y=10 -> hit a t=5 depuis y=15 */
TEST(PlaneTest, OffsetPoint) {
    Plane plane(Point3(0.0f, 10.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), nullptr);

    Ray ray(Point3(0.0f, 15.0f, 0.0f), Vec3(0.0f, -1.0f, 0.0f));
    Interval ray_t(0.001f, INFINITY_F);
    HitRecord rec;

    bool hit = plane.hit(ray, ray_t, rec);
    EXPECT_TRUE(hit == true);
    EXPECT_NEAR(rec.p.y, 10.0f, EPS);
    EXPECT_NEAR(rec.t, 5.0f, EPS);
}
