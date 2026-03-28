/** @file test_hit_record.cpp
 *  @brief Tests de HitRecord (set_face_normal, front/back face) */

#include <gtest/gtest.h>
#include <cmath>
#include "raytracer/core/hit_record.cuh"
#include "raytracer/core/ray.cuh"
#include "raytracer/core/vec3.cuh"

using namespace rt;

constexpr float EPS = 1e-5f;

/** @brief Front face quand ray oppose a la normale sortante */
TEST(HitRecordTest, FrontFaceWhenRayOpposesNormal) {
    Ray r(Point3(0, 0, 0), Vec3(0, 0, -1));
    Vec3 outward_normal(0, 0, 1);

    HitRecord rec;
    rec.set_face_normal(r, outward_normal);

    EXPECT_TRUE(rec.front_face);
    EXPECT_NEAR(rec.normal.x, 0.0f, EPS);
    EXPECT_NEAR(rec.normal.y, 0.0f, EPS);
    EXPECT_NEAR(rec.normal.z, 1.0f, EPS);
}

/** @brief Back face quand ray aligne avec la normale */
TEST(HitRecordTest, BackFaceWhenRayAlignedWithNormal) {
    Ray r(Point3(0, 0, 0), Vec3(0, 0, 1));
    Vec3 outward_normal(0, 0, 1);

    HitRecord rec;
    rec.set_face_normal(r, outward_normal);

    EXPECT_FALSE(rec.front_face);
    EXPECT_NEAR(rec.normal.x, 0.0f, EPS);
    EXPECT_NEAR(rec.normal.y, 0.0f, EPS);
    EXPECT_NEAR(rec.normal.z, -1.0f, EPS);
}

/** @brief Verif flip de la normale en back face */
TEST(HitRecordTest, NormalFlipsForBackFace) {
    Ray r(Point3(0, 0, 0), Vec3(0, 0, 1));
    Vec3 outward_normal(0, 0, 1);

    HitRecord rec;
    rec.set_face_normal(r, outward_normal);

    EXPECT_FALSE(rec.front_face);
    EXPECT_NEAR(rec.normal.x, -outward_normal.x, EPS);
    EXPECT_NEAR(rec.normal.y, -outward_normal.y, EPS);
    EXPECT_NEAR(rec.normal.z, -outward_normal.z, EPS);
}

/** @brief Normale inchangee en front face */
TEST(HitRecordTest, NormalPreservedForFrontFace) {
    Ray r(Point3(0, 0, 0), Vec3(0, 0, -1));
    Vec3 outward_normal(0, 0, 1);

    HitRecord rec;
    rec.set_face_normal(r, outward_normal);

    EXPECT_TRUE(rec.front_face);
    EXPECT_NEAR(rec.normal.x, outward_normal.x, EPS);
    EXPECT_NEAR(rec.normal.y, outward_normal.y, EPS);
    EXPECT_NEAR(rec.normal.z, outward_normal.z, EPS);
}

/** @brief Rayon oblique -> reste front face */
TEST(HitRecordTest, ObliqueRayFrontFace) {
    Ray r(Point3(0, 0, 0), Vec3(1, 0, -1));
    Vec3 outward_normal(0, 0, 1);

    HitRecord rec;
    rec.set_face_normal(r, outward_normal);

    EXPECT_TRUE(rec.front_face);
    EXPECT_NEAR(rec.normal.x, 0.0f, EPS);
    EXPECT_NEAR(rec.normal.y, 0.0f, EPS);
    EXPECT_NEAR(rec.normal.z, 1.0f, EPS);
}
