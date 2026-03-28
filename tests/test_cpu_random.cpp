/** @file test_cpu_random.cpp
 *  @brief Tests CPURandom - bornes, reproductibilite, distributions geometriques */

#include <gtest/gtest.h>
#include <cmath>

#include "raytracer/core/random.cuh"

using namespace rt;

constexpr float EPS = 1e-5f;
constexpr int N = 1000;

/** @brief random_float() dans [0,1) sur 1000 echantillons */
TEST(CPURandomTest, FloatInRange01) {
    CPURandom rng(123);
    for (int i = 0; i < N; ++i) {
        float val = rng.random_float();
        EXPECT_GE(val, 0.0f);
        EXPECT_LT(val, 1.0f);
    }
}

/** @brief random_float(-5,5) -> toujours dans [-5,5) */
TEST(CPURandomTest, FloatInCustomRange) {
    CPURandom rng(456);
    for (int i = 0; i < N; ++i) {
        float val = rng.random_float(-5.0f, 5.0f);
        EXPECT_GE(val, -5.0f);
        EXPECT_LT(val, 5.0f);
    }
}

/** @brief Meme seed -> meme sequence de nb */
TEST(CPURandomTest, SeedReproducibility) {
    CPURandom rng1(42);
    CPURandom rng2(42);
    for (int i = 0; i < 100; ++i) {
        EXPECT_FLOAT_EQ(rng1.random_float(), rng2.random_float());
    }
}

/** @brief Seeds differents -> sequences distinctes */
TEST(CPURandomTest, DifferentSeeds) {
    CPURandom rng0(0);
    CPURandom rng1(1);
    bool found_different = false;
    for (int i = 0; i < 10; ++i) {
        if (rng0.random_float() != rng1.random_float()) {
            found_different = true;
            break;
        }
    }
    EXPECT_TRUE(found_different);
}

/** @brief random_vec3() composantes dans [0,1) */
TEST(CPURandomTest, Vec3ComponentsInRange) {
    CPURandom rng(789);
    for (int i = 0; i < N; ++i) {
        Vec3 v = random_vec3(rng);
        EXPECT_GE(v.x, 0.0f);
        EXPECT_LT(v.x, 1.0f);
        EXPECT_GE(v.y, 0.0f);
        EXPECT_LT(v.y, 1.0f);
        EXPECT_GE(v.z, 0.0f);
        EXPECT_LT(v.z, 1.0f);
    }
}

/** @brief random_vec3(-1,1) -> xyz dans [-1,1) */
TEST(CPURandomTest, Vec3MinMaxRange) {
    CPURandom rng(101);
    for (int i = 0; i < N; ++i) {
        Vec3 v = random_vec3(-1.0f, 1.0f, rng);
        EXPECT_GE(v.x, -1.0f);
        EXPECT_LT(v.x, 1.0f);
        EXPECT_GE(v.y, -1.0f);
        EXPECT_LT(v.y, 1.0f);
        EXPECT_GE(v.z, -1.0f);
        EXPECT_LT(v.z, 1.0f);
    }
}

/** @brief random_unit_vector() -> norme == 1 */
TEST(CPURandomTest, UnitVectorHasUnitLength) {
    CPURandom rng(202);
    for (int i = 0; i < N; ++i) {
        Vec3 v = random_unit_vector(rng);
        EXPECT_NEAR(v.length(), 1.0f, EPS);
    }
}

/** @brief random_in_unit_sphere() -> norme <= 1 */
TEST(CPURandomTest, InUnitSphereNormBound) {
    CPURandom rng(303);
    for (int i = 0; i < N; ++i) {
        Vec3 v = random_in_unit_sphere(rng);
        EXPECT_LE(v.length(), 1.0f + EPS);
    }
}

/** @brief random_on_hemisphere() -> dot(v,n) > 0 toujours */
TEST(CPURandomTest, OnHemisphereSameDirection) {
    CPURandom rng(404);
    Vec3 normal(0.0f, 1.0f, 0.0f);
    for (int i = 0; i < N; ++i) {
        Vec3 v = random_on_hemisphere(normal, rng);
        EXPECT_GT(dot(v, normal), 0.0f);
    }
}

/** @brief random_in_unit_disk() -> z==0 et norme xy <= 1 */
TEST(CPURandomTest, InUnitDiskProperties) {
    CPURandom rng(505);
    for (int i = 0; i < N; ++i) {
        Vec3 v = random_in_unit_disk(rng);
        EXPECT_FLOAT_EQ(v.z, 0.0f);
        float len = std::sqrt(v.x * v.x + v.y * v.y);
        EXPECT_LE(len, 1.0f + EPS);
    }
}

/** @brief random_cosine_direction() -> z >= 0 (hemisphere sup) */
TEST(CPURandomTest, CosineDirectionUpperHemisphere) {
    CPURandom rng(606);
    for (int i = 0; i < N; ++i) {
        Vec3 v = random_cosine_direction(rng);
        EXPECT_GE(v.z, 0.0f);
    }
}
