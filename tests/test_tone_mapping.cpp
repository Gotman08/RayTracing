/**
 * Unit Tests - Tone Mapping & Gamma Correction
 */

#include <iostream>
#include <cmath>

#include "raytracer/core/vec3.cuh"
#include "raytracer/rendering/tone_mapping.cuh"

using namespace rt;

#define TEST_ASSERT(expr) \
    do { \
        if (!(expr)) { \
            std::cerr << "    FAILED: " << #expr << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            return false; \
        } \
    } while(0)

#define TEST_ASSERT_NEAR(a, b, eps) \
    do { \
        if (std::abs((a) - (b)) > (eps)) { \
            std::cerr << "    FAILED: " << #a << " (" << (a) << ") != " << #b << " (" << (b) << ")" \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            return false; \
        } \
    } while(0)

#define RUN_TEST(test_func) \
    do { \
        if (test_func()) { \
            std::cout << "  [PASS] " << #test_func << "\n"; \
            passed++; \
        } else { \
            std::cout << "  [FAIL] " << #test_func << "\n"; \
            failed++; \
        } \
        total++; \
    } while(0)

// ============================================================================
// Reinhard tone mapping tests
// ============================================================================

static bool test_tone_mapping_zero() {
    Color result = apply_tone_mapping(Color(0, 0, 0));
    TEST_ASSERT_NEAR(result.x, 0.0f, 1e-6f);
    TEST_ASSERT_NEAR(result.y, 0.0f, 1e-6f);
    TEST_ASSERT_NEAR(result.z, 0.0f, 1e-6f);
    return true;
}

static bool test_tone_mapping_one() {
    Color result = apply_tone_mapping(Color(1, 1, 1));
    // 1/(1+1) = 0.5
    TEST_ASSERT_NEAR(result.x, 0.5f, 1e-6f);
    TEST_ASSERT_NEAR(result.y, 0.5f, 1e-6f);
    TEST_ASSERT_NEAR(result.z, 0.5f, 1e-6f);
    return true;
}

static bool test_tone_mapping_high_value() {
    // Very bright values should be compressed below 1
    Color result = apply_tone_mapping(Color(100, 100, 100));
    // 100/101 ≈ 0.99
    TEST_ASSERT(result.x < 1.0f);
    TEST_ASSERT(result.x > 0.98f);
    return true;
}

static bool test_tone_mapping_preserves_ratios() {
    Color hdr(2.0f, 4.0f, 8.0f);
    Color result = apply_tone_mapping(hdr);
    // 2/3, 4/5, 8/9
    TEST_ASSERT_NEAR(result.x, 2.0f / 3.0f, 1e-5f);
    TEST_ASSERT_NEAR(result.y, 4.0f / 5.0f, 1e-5f);
    TEST_ASSERT_NEAR(result.z, 8.0f / 9.0f, 1e-5f);
    return true;
}

static bool test_tone_mapping_monotonic() {
    // Brighter input should produce brighter output
    Color dim = apply_tone_mapping(Color(0.5f, 0.5f, 0.5f));
    Color bright = apply_tone_mapping(Color(2.0f, 2.0f, 2.0f));
    TEST_ASSERT(bright.x > dim.x);
    return true;
}

// ============================================================================
// Gamma correction tests
// ============================================================================

static bool test_gamma_correction_zero() {
    Color result = gamma_correct(Color(0, 0, 0));
    TEST_ASSERT_NEAR(result.x, 0.0f, 1e-6f);
    TEST_ASSERT_NEAR(result.y, 0.0f, 1e-6f);
    TEST_ASSERT_NEAR(result.z, 0.0f, 1e-6f);
    return true;
}

static bool test_gamma_correction_one() {
    Color result = gamma_correct(Color(1, 1, 1));
    TEST_ASSERT_NEAR(result.x, 1.0f, 1e-5f);
    TEST_ASSERT_NEAR(result.y, 1.0f, 1e-5f);
    TEST_ASSERT_NEAR(result.z, 1.0f, 1e-5f);
    return true;
}

static bool test_gamma_correction_midtone() {
    // Gamma 2.2: 0.5^(1/2.2) ≈ 0.7297
    Color result = gamma_correct(Color(0.5f, 0.5f, 0.5f), 2.2f);
    float expected = powf(0.5f, 1.0f / 2.2f);
    TEST_ASSERT_NEAR(result.x, expected, 1e-4f);
    return true;
}

static bool test_gamma_correction_brightens_darks() {
    // Gamma correction should brighten mid-tones (sRGB display)
    Color linear(0.25f, 0.25f, 0.25f);
    Color corrected = gamma_correct(linear);
    TEST_ASSERT(corrected.x > linear.x);
    return true;
}

static bool test_gamma_correction_negative_clamped() {
    // Negative values should be clamped to 0 (via fmaxf)
    Color result = gamma_correct(Color(-1.0f, -0.5f, 0.0f));
    TEST_ASSERT_NEAR(result.x, 0.0f, 1e-6f);
    TEST_ASSERT_NEAR(result.y, 0.0f, 1e-6f);
    TEST_ASSERT_NEAR(result.z, 0.0f, 1e-6f);
    return true;
}

static bool test_gamma_2_known_value() {
    // With gamma=2: 0.25^(1/2) = 0.5
    Color result = gamma_correct(Color(0.25f, 0.25f, 0.25f), 2.0f);
    TEST_ASSERT_NEAR(result.x, 0.5f, 1e-5f);
    return true;
}

// ============================================================================
// Test runner
// ============================================================================

void run_tone_mapping_tests(int& passed, int& failed, int& total) {
    RUN_TEST(test_tone_mapping_zero);
    RUN_TEST(test_tone_mapping_one);
    RUN_TEST(test_tone_mapping_high_value);
    RUN_TEST(test_tone_mapping_preserves_ratios);
    RUN_TEST(test_tone_mapping_monotonic);
    RUN_TEST(test_gamma_correction_zero);
    RUN_TEST(test_gamma_correction_one);
    RUN_TEST(test_gamma_correction_midtone);
    RUN_TEST(test_gamma_correction_brightens_darks);
    RUN_TEST(test_gamma_correction_negative_clamped);
    RUN_TEST(test_gamma_2_known_value);
}
