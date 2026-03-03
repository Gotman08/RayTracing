/**
 * Unit Tests - Interval
 */

#include <iostream>
#include <cmath>
#include <cfloat>

#include "raytracer/core/interval.cuh"

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

constexpr float EPS = 1e-6f;

// ==============================================================================
// Construction Tests
// ==============================================================================

bool test_interval_default_constructor() {
    Interval i;
    // Default should be empty (min > max)
    TEST_ASSERT(i.min > i.max);
    return true;
}

bool test_interval_value_constructor() {
    Interval i(1.0f, 5.0f);
    TEST_ASSERT_NEAR(i.min, 1.0f, EPS);
    TEST_ASSERT_NEAR(i.max, 5.0f, EPS);
    return true;
}

bool test_interval_merge_constructor() {
    Interval a(1.0f, 3.0f);
    Interval b(2.0f, 5.0f);
    Interval merged(a, b);
    // Merged should be [1, 5]
    TEST_ASSERT_NEAR(merged.min, 1.0f, EPS);
    TEST_ASSERT_NEAR(merged.max, 5.0f, EPS);
    return true;
}

bool test_interval_merge_disjoint() {
    Interval a(1.0f, 2.0f);
    Interval b(4.0f, 5.0f);
    Interval merged(a, b);
    // Merged should span both: [1, 5]
    TEST_ASSERT_NEAR(merged.min, 1.0f, EPS);
    TEST_ASSERT_NEAR(merged.max, 5.0f, EPS);
    return true;
}

// ==============================================================================
// Size Tests
// ==============================================================================

bool test_interval_size() {
    Interval i(2.0f, 7.0f);
    TEST_ASSERT_NEAR(i.size(), 5.0f, EPS);
    return true;
}

bool test_interval_size_zero() {
    Interval i(3.0f, 3.0f);
    TEST_ASSERT_NEAR(i.size(), 0.0f, EPS);
    return true;
}

bool test_interval_size_negative() {
    Interval i(5.0f, 2.0f);  // Invalid interval
    TEST_ASSERT(i.size() < 0.0f);
    return true;
}

// ==============================================================================
// Contains Tests (inclusive: min <= x <= max)
// ==============================================================================

bool test_interval_contains_inside() {
    Interval i(0.0f, 10.0f);
    TEST_ASSERT(i.contains(5.0f) == true);
    return true;
}

bool test_interval_contains_at_min() {
    Interval i(0.0f, 10.0f);
    TEST_ASSERT(i.contains(0.0f) == true);
    return true;
}

bool test_interval_contains_at_max() {
    Interval i(0.0f, 10.0f);
    TEST_ASSERT(i.contains(10.0f) == true);
    return true;
}

bool test_interval_contains_outside_below() {
    Interval i(0.0f, 10.0f);
    TEST_ASSERT(i.contains(-1.0f) == false);
    return true;
}

bool test_interval_contains_outside_above() {
    Interval i(0.0f, 10.0f);
    TEST_ASSERT(i.contains(11.0f) == false);
    return true;
}

// ==============================================================================
// Surrounds Tests (exclusive: min < x < max)
// ==============================================================================

bool test_interval_surrounds_inside() {
    Interval i(0.0f, 10.0f);
    TEST_ASSERT(i.surrounds(5.0f) == true);
    return true;
}

bool test_interval_surrounds_at_min() {
    Interval i(0.0f, 10.0f);
    // Should NOT surround at boundary
    TEST_ASSERT(i.surrounds(0.0f) == false);
    return true;
}

bool test_interval_surrounds_at_max() {
    Interval i(0.0f, 10.0f);
    // Should NOT surround at boundary
    TEST_ASSERT(i.surrounds(10.0f) == false);
    return true;
}

bool test_interval_surrounds_near_min() {
    Interval i(0.0f, 10.0f);
    TEST_ASSERT(i.surrounds(0.001f) == true);
    return true;
}

// ==============================================================================
// Clamp Tests
// ==============================================================================

bool test_interval_clamp_inside() {
    Interval i(0.0f, 10.0f);
    TEST_ASSERT_NEAR(i.clamp(5.0f), 5.0f, EPS);
    return true;
}

bool test_interval_clamp_below() {
    Interval i(0.0f, 10.0f);
    TEST_ASSERT_NEAR(i.clamp(-5.0f), 0.0f, EPS);
    return true;
}

bool test_interval_clamp_above() {
    Interval i(0.0f, 10.0f);
    TEST_ASSERT_NEAR(i.clamp(15.0f), 10.0f, EPS);
    return true;
}

bool test_interval_clamp_at_boundary() {
    Interval i(0.0f, 10.0f);
    TEST_ASSERT_NEAR(i.clamp(0.0f), 0.0f, EPS);
    TEST_ASSERT_NEAR(i.clamp(10.0f), 10.0f, EPS);
    return true;
}

// ==============================================================================
// Expand Tests
// ==============================================================================

bool test_interval_expand() {
    Interval i(2.0f, 8.0f);
    Interval expanded = i.expand(2.0f);
    // Expand by 2 means add 1 to each side
    TEST_ASSERT_NEAR(expanded.min, 1.0f, EPS);
    TEST_ASSERT_NEAR(expanded.max, 9.0f, EPS);
    return true;
}

bool test_interval_expand_zero() {
    Interval i(2.0f, 8.0f);
    Interval expanded = i.expand(0.0f);
    TEST_ASSERT_NEAR(expanded.min, 2.0f, EPS);
    TEST_ASSERT_NEAR(expanded.max, 8.0f, EPS);
    return true;
}

bool test_interval_expand_negative() {
    Interval i(0.0f, 10.0f);
    Interval shrunk = i.expand(-4.0f);
    // Shrink by 4 means subtract 2 from each side
    TEST_ASSERT_NEAR(shrunk.min, 2.0f, EPS);
    TEST_ASSERT_NEAR(shrunk.max, 8.0f, EPS);
    return true;
}

// ==============================================================================
// Test Suite Runner
// ==============================================================================

void run_interval_tests(int& passed, int& failed, int& total) {
    // Construction
    RUN_TEST(test_interval_default_constructor);
    RUN_TEST(test_interval_value_constructor);
    RUN_TEST(test_interval_merge_constructor);
    RUN_TEST(test_interval_merge_disjoint);

    // Size
    RUN_TEST(test_interval_size);
    RUN_TEST(test_interval_size_zero);
    RUN_TEST(test_interval_size_negative);

    // Contains
    RUN_TEST(test_interval_contains_inside);
    RUN_TEST(test_interval_contains_at_min);
    RUN_TEST(test_interval_contains_at_max);
    RUN_TEST(test_interval_contains_outside_below);
    RUN_TEST(test_interval_contains_outside_above);

    // Surrounds
    RUN_TEST(test_interval_surrounds_inside);
    RUN_TEST(test_interval_surrounds_at_min);
    RUN_TEST(test_interval_surrounds_at_max);
    RUN_TEST(test_interval_surrounds_near_min);

    // Clamp
    RUN_TEST(test_interval_clamp_inside);
    RUN_TEST(test_interval_clamp_below);
    RUN_TEST(test_interval_clamp_above);
    RUN_TEST(test_interval_clamp_at_boundary);

    // Expand
    RUN_TEST(test_interval_expand);
    RUN_TEST(test_interval_expand_zero);
    RUN_TEST(test_interval_expand_negative);
}
