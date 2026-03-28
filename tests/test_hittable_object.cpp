/** @file test_hittable_object.cpp
 *  @brief Tests HittableObject - dispatch sphere/plane via union */

#include <gtest/gtest.h>
#include <cmath>

#include "raytracer/geometry/hittable_list.cuh"
#include "raytracer/materials/material.cuh"

using namespace rt;

constexpr float EPS = 1e-5f;

/** @brief Type par defaut == SPHERE */
TEST(HittableObjectTest, DefaultTypeIsSphere) {
    HittableObject obj;
    EXPECT_EQ(obj.type, HittableType::SPHERE);
}

/** @brief Dispatch sphere -> hit a t=4, z=-1 */
TEST(HittableObjectTest, SphereDispatch) {
    Material mat(MaterialType::LAMBERTIAN, Color(1.0f, 0.0f, 0.0f));

    HittableObject obj;
    obj.type = HittableType::SPHERE;
    obj.data.sphere = Sphere(Point3(0.0f, 0.0f, 0.0f), 1.0f, &mat);
    obj.bbox = obj.data.sphere.bounding_box();

    Ray r(Point3(0.0f, 0.0f, -5.0f), Vec3(0.0f, 0.0f, 1.0f));
    HitRecord rec;
    bool result = obj.hit(r, Interval(0.001f, 1e30f), rec);
    EXPECT_TRUE(result);
    EXPECT_NEAR(rec.t, 4.0f, EPS);
    EXPECT_NEAR(rec.p.z, -1.0f, EPS);
}

/** @brief Dispatch plane -> hit a t=5, y=0 */
TEST(HittableObjectTest, PlaneDispatch) {
    Material mat(MaterialType::LAMBERTIAN, Color(0.0f, 1.0f, 0.0f));

    HittableObject obj;
    obj.type = HittableType::PLANE;
    obj.data.plane = Plane(Point3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), &mat);
    obj.bbox = obj.data.plane.bounding_box();

    Ray r(Point3(0.0f, 5.0f, 0.0f), Vec3(0.0f, -1.0f, 0.0f));
    HitRecord rec;
    bool result = obj.hit(r, Interval(0.001f, 1e30f), rec);
    EXPECT_TRUE(result);
    EXPECT_NEAR(rec.t, 5.0f, EPS);
    EXPECT_NEAR(rec.p.y, 0.0f, EPS);
}

/** @brief Ray qui rate la sphere -> hit() false */
TEST(HittableObjectTest, SphereMiss) {
    Material mat(MaterialType::LAMBERTIAN, Color(1.0f, 0.0f, 0.0f));

    HittableObject obj;
    obj.type = HittableType::SPHERE;
    obj.data.sphere = Sphere(Point3(0.0f, 0.0f, 0.0f), 1.0f, &mat);
    obj.bbox = obj.data.sphere.bounding_box();

    Ray r(Point3(0.0f, 5.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f));
    HitRecord rec;
    bool result = obj.hit(r, Interval(0.001f, 1e30f), rec);
    EXPECT_FALSE(result);
}

/** @brief Bbox sphere r=1 -> bornes [-1,1] sur chaque axe */
TEST(HittableObjectTest, BboxSetCorrectly) {
    Material mat(MaterialType::LAMBERTIAN, Color(1.0f, 0.0f, 0.0f));

    HittableObject obj;
    obj.type = HittableType::SPHERE;
    obj.data.sphere = Sphere(Point3(0.0f, 0.0f, 0.0f), 1.0f, &mat);
    obj.bbox = obj.data.sphere.bounding_box();

    EXPECT_LE(obj.bbox.x.min, -1.0f);
    EXPECT_GE(obj.bbox.x.max, 1.0f);
    EXPECT_LE(obj.bbox.y.min, -1.0f);
    EXPECT_GE(obj.bbox.y.max, 1.0f);
    EXPECT_LE(obj.bbox.z.min, -1.0f);
    EXPECT_GE(obj.bbox.z.max, 1.0f);
}
