/** @file test_geometry_utils.cpp
 *  @brief Tests get_sphere_uv() - coord UV sur sphere unitaire */

#include <gtest/gtest.h>
#include <cmath>

#include "raytracer/geometry/geometry_utils.cuh"

using namespace rt;

constexpr float EPS = 1e-5f;

/** @brief Pole nord -> u=0.5, v=0 */
TEST(GeometryUtilsTest, NorthPole) {
    Vec3 p(0.0f, -1.0f, 0.0f);
    float u, v;
    get_sphere_uv(p, u, v);
    EXPECT_NEAR(u, 0.5f, EPS);
    EXPECT_NEAR(v, 0.0f, EPS);
}

/** @brief Pole sud -> u=0.5, v=1 */
TEST(GeometryUtilsTest, SouthPole) {
    Vec3 p(0.0f, 1.0f, 0.0f);
    float u, v;
    get_sphere_uv(p, u, v);
    EXPECT_NEAR(u, 0.5f, EPS);
    EXPECT_NEAR(v, 1.0f, EPS);
}

/** @brief Equateur +X -> v=0.5, u via atan2 */
TEST(GeometryUtilsTest, EquatorPosX) {
    Vec3 p(1.0f, 0.0f, 0.0f);
    float u, v;
    get_sphere_uv(p, u, v);
    float expected_u = (std::atan2(0.0f, 1.0f) + PI) / (2.0f * PI);
    EXPECT_NEAR(u, expected_u, EPS);
    EXPECT_NEAR(v, 0.5f, EPS);
}

/** @brief Equateur -Z -> v=0.5, u correct */
TEST(GeometryUtilsTest, EquatorNegZ) {
    Vec3 p(0.0f, 0.0f, -1.0f);
    float u, v;
    get_sphere_uv(p, u, v);
    float expected_u = (std::atan2(1.0f, 0.0f) + PI) / (2.0f * PI);
    EXPECT_NEAR(u, expected_u, EPS);
    EXPECT_NEAR(v, 0.5f, EPS);
}

/** @brief Equateur +Z -> v=0.5, u symetrique */
TEST(GeometryUtilsTest, EquatorPosZ) {
    Vec3 p(0.0f, 0.0f, 1.0f);
    float u, v;
    get_sphere_uv(p, u, v);
    float expected_u = (std::atan2(-1.0f, 0.0f) + PI) / (2.0f * PI);
    EXPECT_NEAR(u, expected_u, EPS);
    EXPECT_NEAR(v, 0.5f, EPS);
}

/** @brief Balayage 50x50 angles -> u,v toujours dans [0,1] */
TEST(GeometryUtilsTest, UVAlwaysInRange) {
    const int N_theta = 50;
    const int N_phi = 50;
    for (int i = 0; i < N_theta; ++i) {
        float theta = PI * static_cast<float>(i) / static_cast<float>(N_theta - 1);
        for (int j = 0; j < N_phi; ++j) {
            float phi = 2.0f * PI * static_cast<float>(j) / static_cast<float>(N_phi - 1);
            float px = std::sin(theta) * std::cos(phi);
            float py = std::cos(theta);
            float pz = std::sin(theta) * std::sin(phi);
            Vec3 p(px, py, pz);
            float u, v;
            get_sphere_uv(p, u, v);
            EXPECT_GE(u, 0.0f);
            EXPECT_LE(u, 1.0f);
            EXPECT_GE(v, 0.0f);
            EXPECT_LE(v, 1.0f);
        }
    }
}
