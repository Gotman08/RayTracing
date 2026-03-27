/**
 * @file test_tone_mapping.cpp
 * @brief Tests Reinhard tone mapping + correction gamma
 */

#include <gtest/gtest.h>
#include <cmath>

#include "raytracer/core/vec3.cuh"
#include "raytracer/rendering/tone_mapping.cuh"

using namespace rt;

/** @brief Reinhard(0) = 0 */
TEST(ToneMappingTest, Zero) {
    Color result = apply_tone_mapping(Color(0, 0, 0));
    EXPECT_NEAR(result.x, 0.0f, 1e-6f);
    EXPECT_NEAR(result.y, 0.0f, 1e-6f);
    EXPECT_NEAR(result.z, 0.0f, 1e-6f);
}

/** @brief Reinhard(1) = 0.5 */
TEST(ToneMappingTest, One) {
    Color result = apply_tone_mapping(Color(1, 1, 1));
    EXPECT_NEAR(result.x, 0.5f, 1e-6f);
    EXPECT_NEAR(result.y, 0.5f, 1e-6f);
    EXPECT_NEAR(result.z, 0.5f, 1e-6f);
}

/** @brief Reinhard(100) ~ 0.99, reste < 1 */
TEST(ToneMappingTest, HighValue) {
    Color result = apply_tone_mapping(Color(100, 100, 100));
    EXPECT_TRUE(result.x < 1.0f);
    EXPECT_TRUE(result.x > 0.98f);
}

/** @brief Reinhard (2,4,8) -> (2/3, 4/5, 8/9) exact */
TEST(ToneMappingTest, PreservesRatios) {
    Color hdr(2.0f, 4.0f, 8.0f);
    Color result = apply_tone_mapping(hdr);
    EXPECT_NEAR(result.x, 2.0f / 3.0f, 1e-5f);
    EXPECT_NEAR(result.y, 4.0f / 5.0f, 1e-5f);
    EXPECT_NEAR(result.z, 8.0f / 9.0f, 1e-5f);
}

/** @brief Monotonie : bright > dim apres tone map */
TEST(ToneMappingTest, Monotonic) {
    Color dim = apply_tone_mapping(Color(0.5f, 0.5f, 0.5f));
    Color bright = apply_tone_mapping(Color(2.0f, 2.0f, 2.0f));
    EXPECT_TRUE(bright.x > dim.x);
}


/** @brief Gamma(0) = 0 quel que soit gamma */
TEST(ToneMappingTest, GammaCorrectionZero) {
    Color result = gamma_correct(Color(0, 0, 0));
    EXPECT_NEAR(result.x, 0.0f, 1e-6f);
    EXPECT_NEAR(result.y, 0.0f, 1e-6f);
    EXPECT_NEAR(result.z, 0.0f, 1e-6f);
}

/** @brief Gamma(1) = 1 invariant */
TEST(ToneMappingTest, GammaCorrectionOne) {
    Color result = gamma_correct(Color(1, 1, 1));
    EXPECT_NEAR(result.x, 1.0f, 1e-5f);
    EXPECT_NEAR(result.y, 1.0f, 1e-5f);
    EXPECT_NEAR(result.z, 1.0f, 1e-5f);
}

/** @brief Gamma 2.2 midtone : 0.5^(1/2.2) ~ 0.73 */
TEST(ToneMappingTest, GammaCorrectionMidtone) {
    Color result = gamma_correct(Color(0.5f, 0.5f, 0.5f), 2.2f);
    float expected = powf(0.5f, 1.0f / 2.2f);
    EXPECT_NEAR(result.x, expected, 1e-4f);
}

/** @brief Gamma eclaircit les sombres : corrected > linear */
TEST(ToneMappingTest, GammaCorrectionBrightensDarks) {
    Color linear(0.25f, 0.25f, 0.25f);
    Color corrected = gamma_correct(linear);
    EXPECT_TRUE(corrected.x > linear.x);
}

/** @brief Valeurs negatives -> clamp a 0 avant gamma */
TEST(ToneMappingTest, GammaCorrectionNegativeClamped) {
    Color result = gamma_correct(Color(-1.0f, -0.5f, 0.0f));
    EXPECT_NEAR(result.x, 0.0f, 1e-6f);
    EXPECT_NEAR(result.y, 0.0f, 1e-6f);
    EXPECT_NEAR(result.z, 0.0f, 1e-6f);
}

/** @brief Gamma 2.0 -> sqrt : 0.25^(1/2) = 0.5 */
TEST(ToneMappingTest, Gamma2KnownValue) {
    Color result = gamma_correct(Color(0.25f, 0.25f, 0.25f), 2.0f);
    EXPECT_NEAR(result.x, 0.5f, 1e-5f);
}
