/** @file test_sky.cpp
 *  @brief Tests du modele de ciel (gradient, soleil, sky_black) */

#include <gtest/gtest.h>
#include <cmath>
#include "raytracer/environment/sky.cuh"
#include "raytracer/core/vec3.cuh"

using namespace rt;

constexpr float EPS = 1e-5f;

/** @brief Ctor defaut -> horizon blanc */
TEST(SkyTest, DefaultHorizonColor) {
    Sky sky;
    EXPECT_NEAR(sky.horizon_color.x, 1.0f, EPS);
    EXPECT_NEAR(sky.horizon_color.y, 1.0f, EPS);
    EXPECT_NEAR(sky.horizon_color.z, 1.0f, EPS);
}

/** @brief Ctor defaut -> zenith bleu clair */
TEST(SkyTest, DefaultZenithColor) {
    Sky sky;
    EXPECT_NEAR(sky.zenith_color.x, 0.5f, EPS);
    EXPECT_NEAR(sky.zenith_color.y, 0.7f, EPS);
    EXPECT_NEAR(sky.zenith_color.z, 1.0f, EPS);
}

/** @brief Pas de soleil par defaut (intensite 0) */
TEST(SkyTest, DefaultNoSun) {
    Sky sky;
    EXPECT_FLOAT_EQ(sky.sun_intensity, 0.0f);
}

/** @brief Ctor custom -> horizon et zenith corrects */
TEST(SkyTest, CustomColors) {
    Color h(0.1f, 0.2f, 0.3f);
    Color z(0.4f, 0.5f, 0.6f);
    Sky sky(h, z);

    EXPECT_NEAR(sky.horizon_color.x, 0.1f, EPS);
    EXPECT_NEAR(sky.horizon_color.y, 0.2f, EPS);
    EXPECT_NEAR(sky.horizon_color.z, 0.3f, EPS);
    EXPECT_NEAR(sky.zenith_color.x, 0.4f, EPS);
    EXPECT_NEAR(sky.zenith_color.y, 0.5f, EPS);
    EXPECT_NEAR(sky.zenith_color.z, 0.6f, EPS);
}

/** @brief Gradient au zenith -> doit rendre zenith_color */
TEST(SkyTest, GradientAtZenith) {
    Sky sky;
    Color c = sky.get_color(Vec3(0, 1, 0));

    EXPECT_NEAR(c.x, sky.zenith_color.x, EPS);
    EXPECT_NEAR(c.y, sky.zenith_color.y, EPS);
    EXPECT_NEAR(c.z, sky.zenith_color.z, EPS);
}

/** @brief Gradient a l'horizon -> horizon_color */
TEST(SkyTest, GradientAtHorizon) {
    Sky sky;
    Color c = sky.get_color(Vec3(0, -1, 0));

    EXPECT_NEAR(c.x, sky.horizon_color.x, EPS);
    EXPECT_NEAR(c.y, sky.horizon_color.y, EPS);
    EXPECT_NEAR(c.z, sky.horizon_color.z, EPS);
}

/** @brief Dir horizontale -> mix 50/50 horizon+zenith */
TEST(SkyTest, GradientAtMiddle) {
    Sky sky;
    Color c = sky.get_color(Vec3(1, 0, 0));

    Color expected = 0.5f * sky.horizon_color + 0.5f * sky.zenith_color;
    EXPECT_NEAR(c.x, expected.x, EPS);
    EXPECT_NEAR(c.y, expected.y, EPS);
    EXPECT_NEAR(c.z, expected.z, EPS);
}

/** @brief set_sun() normalise la dir et stocke l'intensite */
TEST(SkyTest, SetSunNormalizesDirection) {
    Sky sky;
    sky.set_sun(Vec3(1, 1, 0), 5.0f);

    float len = sky.sun_direction.length();
    EXPECT_NEAR(len, 1.0f, EPS);
    EXPECT_FLOAT_EQ(sky.sun_intensity, 5.0f);
}

/** @brief Soleil actif -> pixel plus lumineux qu'avant */
TEST(SkyTest, SunAddsIntensity) {
    Sky sky;

    Color no_sun = sky.get_color(Vec3(0, 1, 0));

    sky.set_sun(Vec3(0, 1, 0), 10.0f, 0.05f);

    Color with_sun = sky.get_color(Vec3(0, 1, 0));

    bool brighter = (with_sun.x > no_sun.x + EPS) ||
                    (with_sun.y > no_sun.y + EPS) ||
                    (with_sun.z > no_sun.z + EPS);
    EXPECT_TRUE(brighter);
}

/** @brief Fn libre sky_gradient() -> meme resultat que Sky par defaut */
TEST(SkyTest, SkyGradientFunction) {
    Color c = sky_gradient(Vec3(0, 1, 0));

    EXPECT_NEAR(c.x, 0.5f, EPS);
    EXPECT_NEAR(c.y, 0.7f, EPS);
    EXPECT_NEAR(c.z, 1.0f, EPS);
}

/** @brief sky_black() -> toujours noir peu importe la dir */
TEST(SkyTest, SkyBlackFunction) {
    Color c = sky_black(Vec3(0, 1, 0));

    EXPECT_NEAR(c.x, 0.0f, EPS);
    EXPECT_NEAR(c.y, 0.0f, EPS);
    EXPECT_NEAR(c.z, 0.0f, EPS);

    Color c2 = sky_black(Vec3(1, -1, 0.5f));
    EXPECT_NEAR(c2.x, 0.0f, EPS);
    EXPECT_NEAR(c2.y, 0.0f, EPS);
    EXPECT_NEAR(c2.z, 0.0f, EPS);
}
