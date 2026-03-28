/** @file test_hittable_list.cpp
 *  @brief Tests HittableList - ajout, bbox, closest hit, overflow */

#include <gtest/gtest.h>
#include "raytracer/geometry/hittable_list.cuh"
#include "raytracer/materials/material.cuh"

using namespace rt;

/** @brief Fixture - tableau de 10 HittableObject + mat par defaut */
class HittableListTest : public ::testing::Test {
protected:
    HittableObject objects[10];
    HittableList   list;
    Material       mat;

    void SetUp() override {
        list.objects  = objects;
        list.count    = 0;
        list.capacity = 10;
        mat = Material();
    }
};

/** @brief Liste vide -> hit() retourne false */
TEST_F(HittableListTest, EmptyListHitReturnsFalse) {
    EXPECT_EQ(list.count, 0);

    Ray       r(Point3(0, 0, 0), Vec3(0, 0, -1));
    HitRecord rec;
    EXPECT_FALSE(list.hit(r, Interval(0.001f, 1e30f), rec));
}

/** @brief add_sphere() incremente count a 1 */
TEST_F(HittableListTest, AddSphere) {
    list.add_sphere(Point3(0, 0, 0), 1.0f, &mat);
    EXPECT_EQ(list.count, 1);
}

/** @brief add_plane() incremente count a 1 */
TEST_F(HittableListTest, AddPlane) {
    list.add_plane(Point3(0, 0, 0), Vec3(0, 1, 0), &mat);
    EXPECT_EQ(list.count, 1);
}

/** @brief 3 spheres + 1 plan -> count==4 */
TEST_F(HittableListTest, AddMultiple) {
    list.add_sphere(Point3(0, 0, -2), 1.0f, &mat);
    list.add_sphere(Point3(3, 0, -2), 0.5f, &mat);
    list.add_sphere(Point3(-3, 0, -2), 0.5f, &mat);
    list.add_plane(Point3(0, -1, 0), Vec3(0, 1, 0), &mat);
    EXPECT_EQ(list.count, 4);
}

/** @brief 2 spheres ecartees -> bbox englobe [-6,6] en x */
TEST_F(HittableListTest, BboxExpansion) {
    list.add_sphere(Point3(-5, 0, 0), 1.0f, &mat);
    list.add_sphere(Point3(5, 0, 0), 1.0f, &mat);

    const AABB& bb = list.bounding_box();
    EXPECT_LE(bb.x.min, -6.0f);
    EXPECT_GE(bb.x.max, 6.0f);
}

/** @brief Overflow capacite -> count ne depasse pas capacity */
TEST_F(HittableListTest, CapacityOverflow) {
    list.capacity = 2;

    list.add_sphere(Point3(0, 0, 0), 1.0f, &mat);
    list.add_sphere(Point3(1, 0, 0), 1.0f, &mat);
    list.add_sphere(Point3(2, 0, 0), 1.0f, &mat);

    EXPECT_EQ(list.count, 2);
}

/** @brief 2 spheres sur Z, hit() retourne la plus proche */
TEST_F(HittableListTest, ClosestHitSelection) {
    list.add_sphere(Point3(0, 0, -3), 1.0f, &mat);
    list.add_sphere(Point3(0, 0, -5), 1.0f, &mat);

    Ray       r(Point3(0, 0, 0), Vec3(0, 0, -1));
    HitRecord rec;
    bool      did_hit = list.hit(r, Interval(0.001f, 1e30f), rec);

    EXPECT_TRUE(did_hit);
    EXPECT_NEAR(rec.t, 2.0f, 0.01f);
}

/** @brief Ray vers +Y mais sphere en -Y -> miss */
TEST_F(HittableListTest, HitMiss) {
    list.add_sphere(Point3(0, -10, 0), 1.0f, &mat);

    Ray       r(Point3(0, 0, 0), Vec3(0, 1, 0));
    HitRecord rec;
    EXPECT_FALSE(list.hit(r, Interval(0.001f, 1e30f), rec));
}
