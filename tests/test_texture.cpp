/** @file test_texture.cpp
 *  @brief Tests Texture & SolidColor (ctor, value() constante) */

#include <gtest/gtest.h>
#include "raytracer/textures/texture.cuh"
#include "raytracer/textures/solid_color.cuh"
#include "raytracer/core/vec3.cuh"

using namespace rt;

constexpr float EPS = 1e-5f;

/** @brief Ctor par defaut -> blanc (1,1,1) */
TEST(TextureTest, DefaultIsWhite) {
    Texture tex;
    EXPECT_NEAR(tex.color.x, 1.0f, EPS);
    EXPECT_NEAR(tex.color.y, 1.0f, EPS);
    EXPECT_NEAR(tex.color.z, 1.0f, EPS);
}

/** @brief Ctor avec couleur custom -> stockee correctement */
TEST(TextureTest, CustomColor) {
    Texture tex(Color(0.5f, 0.3f, 0.1f));
    EXPECT_NEAR(tex.color.x, 0.5f, EPS);
    EXPECT_NEAR(tex.color.y, 0.3f, EPS);
    EXPECT_NEAR(tex.color.z, 0.1f, EPS);
}

/** @brief value() ignore les UV et la position */
TEST(TextureTest, ValueIgnoresUVAndPosition) {
    Texture tex(Color(0.2f, 0.4f, 0.6f));
    Point3 p(1, 2, 3);
    Point3 q(10, 20, 30);

    Color c1 = tex.value(0.0f, 0.0f, p);
    Color c2 = tex.value(0.5f, 0.5f, q);

    EXPECT_NEAR(c1.x, c2.x, EPS);
    EXPECT_NEAR(c1.y, c2.y, EPS);
    EXPECT_NEAR(c1.z, c2.z, EPS);
}

/** @brief SolidColor par defaut -> noir (0,0,0) */
TEST(SolidColorTest, DefaultIsBlack) {
    SolidColor sc;
    EXPECT_NEAR(sc.albedo.x, 0.0f, EPS);
    EXPECT_NEAR(sc.albedo.y, 0.0f, EPS);
    EXPECT_NEAR(sc.albedo.z, 0.0f, EPS);
}

/** @brief Init depuis un Color -> albedo OK */
TEST(SolidColorTest, FromColor) {
    SolidColor sc(Color(1.0f, 0.0f, 0.0f));
    EXPECT_NEAR(sc.albedo.x, 1.0f, EPS);
    EXPECT_NEAR(sc.albedo.y, 0.0f, EPS);
    EXPECT_NEAR(sc.albedo.z, 0.0f, EPS);
}

/** @brief Init depuis 3 floats RGB */
TEST(SolidColorTest, FromRGB) {
    SolidColor sc(0.2f, 0.4f, 0.6f);
    EXPECT_NEAR(sc.albedo.x, 0.2f, EPS);
    EXPECT_NEAR(sc.albedo.y, 0.4f, EPS);
    EXPECT_NEAR(sc.albedo.z, 0.6f, EPS);
}

/** @brief value() constante quelle que soit la pos */
TEST(SolidColorTest, ValueIsConstant) {
    SolidColor sc(0.3f, 0.5f, 0.7f);
    Point3 p1(0, 0, 0);
    Point3 p2(100, 200, 300);

    Color c1 = sc.value(0.0f, 0.0f, p1);
    Color c2 = sc.value(0.9f, 0.9f, p2);

    EXPECT_NEAR(c1.x, c2.x, EPS);
    EXPECT_NEAR(c1.y, c2.y, EPS);
    EXPECT_NEAR(c1.z, c2.z, EPS);
}
