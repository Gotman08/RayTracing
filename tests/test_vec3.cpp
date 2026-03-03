/**
 * Unit Tests - Vec3 / Point3 / Color
 */

#include <iostream>
#include <cmath>

// Include Vec3 (uses __host__ __device__ which works on host)
#include "raytracer/core/vec3.cuh"

using namespace rt;

// Test macros from test_main.cpp
extern int passed, failed, total;

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

bool test_vec3_default_constructor() {
    Vec3 v;
    TEST_ASSERT_NEAR(v.x, 0.0f, EPS);
    TEST_ASSERT_NEAR(v.y, 0.0f, EPS);
    TEST_ASSERT_NEAR(v.z, 0.0f, EPS);
    return true;
}

bool test_vec3_value_constructor() {
    Vec3 v(1.0f, 2.0f, 3.0f);
    TEST_ASSERT_NEAR(v.x, 1.0f, EPS);
    TEST_ASSERT_NEAR(v.y, 2.0f, EPS);
    TEST_ASSERT_NEAR(v.z, 3.0f, EPS);
    return true;
}

bool test_vec3_single_value_constructor() {
    Vec3 v(5.0f);
    TEST_ASSERT_NEAR(v.x, 5.0f, EPS);
    TEST_ASSERT_NEAR(v.y, 5.0f, EPS);
    TEST_ASSERT_NEAR(v.z, 5.0f, EPS);
    return true;
}

// ==============================================================================
// Operator Tests
// ==============================================================================

bool test_vec3_negation() {
    Vec3 v(1.0f, -2.0f, 3.0f);
    Vec3 neg = -v;
    TEST_ASSERT_NEAR(neg.x, -1.0f, EPS);
    TEST_ASSERT_NEAR(neg.y, 2.0f, EPS);
    TEST_ASSERT_NEAR(neg.z, -3.0f, EPS);
    return true;
}

bool test_vec3_addition() {
    Vec3 a(1.0f, 2.0f, 3.0f);
    Vec3 b(4.0f, 5.0f, 6.0f);
    Vec3 c = a + b;
    TEST_ASSERT_NEAR(c.x, 5.0f, EPS);
    TEST_ASSERT_NEAR(c.y, 7.0f, EPS);
    TEST_ASSERT_NEAR(c.z, 9.0f, EPS);
    return true;
}

bool test_vec3_subtraction() {
    Vec3 a(4.0f, 5.0f, 6.0f);
    Vec3 b(1.0f, 2.0f, 3.0f);
    Vec3 c = a - b;
    TEST_ASSERT_NEAR(c.x, 3.0f, EPS);
    TEST_ASSERT_NEAR(c.y, 3.0f, EPS);
    TEST_ASSERT_NEAR(c.z, 3.0f, EPS);
    return true;
}

bool test_vec3_multiplication() {
    Vec3 a(1.0f, 2.0f, 3.0f);
    Vec3 b(2.0f, 3.0f, 4.0f);
    Vec3 c = a * b;
    TEST_ASSERT_NEAR(c.x, 2.0f, EPS);
    TEST_ASSERT_NEAR(c.y, 6.0f, EPS);
    TEST_ASSERT_NEAR(c.z, 12.0f, EPS);
    return true;
}

bool test_vec3_scalar_multiplication() {
    Vec3 v(1.0f, 2.0f, 3.0f);
    Vec3 c1 = v * 2.0f;
    Vec3 c2 = 2.0f * v;
    TEST_ASSERT_NEAR(c1.x, 2.0f, EPS);
    TEST_ASSERT_NEAR(c1.y, 4.0f, EPS);
    TEST_ASSERT_NEAR(c1.z, 6.0f, EPS);
    TEST_ASSERT_NEAR(c2.x, 2.0f, EPS);
    TEST_ASSERT_NEAR(c2.y, 4.0f, EPS);
    TEST_ASSERT_NEAR(c2.z, 6.0f, EPS);
    return true;
}

bool test_vec3_division() {
    Vec3 v(2.0f, 4.0f, 6.0f);
    Vec3 c = v / 2.0f;
    TEST_ASSERT_NEAR(c.x, 1.0f, EPS);
    TEST_ASSERT_NEAR(c.y, 2.0f, EPS);
    TEST_ASSERT_NEAR(c.z, 3.0f, EPS);
    return true;
}

bool test_vec3_index_operator() {
    Vec3 v(1.0f, 2.0f, 3.0f);
    TEST_ASSERT_NEAR(v[0], 1.0f, EPS);
    TEST_ASSERT_NEAR(v[1], 2.0f, EPS);
    TEST_ASSERT_NEAR(v[2], 3.0f, EPS);
    return true;
}

bool test_vec3_compound_addition() {
    Vec3 v(1.0f, 2.0f, 3.0f);
    v += Vec3(1.0f, 1.0f, 1.0f);
    TEST_ASSERT_NEAR(v.x, 2.0f, EPS);
    TEST_ASSERT_NEAR(v.y, 3.0f, EPS);
    TEST_ASSERT_NEAR(v.z, 4.0f, EPS);
    return true;
}

bool test_vec3_compound_multiplication() {
    Vec3 v(1.0f, 2.0f, 3.0f);
    v *= 2.0f;
    TEST_ASSERT_NEAR(v.x, 2.0f, EPS);
    TEST_ASSERT_NEAR(v.y, 4.0f, EPS);
    TEST_ASSERT_NEAR(v.z, 6.0f, EPS);
    return true;
}

bool test_vec3_compound_division() {
    Vec3 v(2.0f, 4.0f, 6.0f);
    v /= 2.0f;
    TEST_ASSERT_NEAR(v.x, 1.0f, EPS);
    TEST_ASSERT_NEAR(v.y, 2.0f, EPS);
    TEST_ASSERT_NEAR(v.z, 3.0f, EPS);
    return true;
}

// ==============================================================================
// Method Tests
// ==============================================================================

bool test_vec3_length() {
    Vec3 v(3.0f, 4.0f, 0.0f);
    TEST_ASSERT_NEAR(v.length(), 5.0f, EPS);
    return true;
}

bool test_vec3_length_squared() {
    Vec3 v(3.0f, 4.0f, 0.0f);
    TEST_ASSERT_NEAR(v.length_squared(), 25.0f, EPS);
    return true;
}

bool test_vec3_normalized() {
    Vec3 v(3.0f, 4.0f, 0.0f);
    Vec3 n = v.normalized();
    TEST_ASSERT_NEAR(n.length(), 1.0f, EPS);
    TEST_ASSERT_NEAR(n.x, 0.6f, EPS);
    TEST_ASSERT_NEAR(n.y, 0.8f, EPS);
    TEST_ASSERT_NEAR(n.z, 0.0f, EPS);
    return true;
}

bool test_vec3_near_zero() {
    Vec3 small(1e-9f, 1e-9f, 1e-9f);
    Vec3 large(1.0f, 0.0f, 0.0f);
    TEST_ASSERT(small.near_zero() == true);
    TEST_ASSERT(large.near_zero() == false);
    return true;
}

// ==============================================================================
// Free Function Tests
// ==============================================================================

bool test_vec3_dot() {
    Vec3 a(1.0f, 2.0f, 3.0f);
    Vec3 b(4.0f, 5.0f, 6.0f);
    float d = dot(a, b);
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    TEST_ASSERT_NEAR(d, 32.0f, EPS);
    return true;
}

bool test_vec3_cross() {
    Vec3 a(1.0f, 0.0f, 0.0f);
    Vec3 b(0.0f, 1.0f, 0.0f);
    Vec3 c = cross(a, b);
    // i x j = k
    TEST_ASSERT_NEAR(c.x, 0.0f, EPS);
    TEST_ASSERT_NEAR(c.y, 0.0f, EPS);
    TEST_ASSERT_NEAR(c.z, 1.0f, EPS);
    return true;
}

bool test_vec3_cross_anticommutative() {
    Vec3 a(1.0f, 0.0f, 0.0f);
    Vec3 b(0.0f, 1.0f, 0.0f);
    Vec3 c1 = cross(a, b);
    Vec3 c2 = cross(b, a);
    // a x b = -(b x a)
    TEST_ASSERT_NEAR(c1.x, -c2.x, EPS);
    TEST_ASSERT_NEAR(c1.y, -c2.y, EPS);
    TEST_ASSERT_NEAR(c1.z, -c2.z, EPS);
    return true;
}

bool test_vec3_reflect() {
    // Reflect (1, -1, 0) off horizontal surface with normal (0, 1, 0)
    Vec3 v(1.0f, -1.0f, 0.0f);
    Vec3 n(0.0f, 1.0f, 0.0f);
    Vec3 r = reflect(v, n);
    TEST_ASSERT_NEAR(r.x, 1.0f, EPS);
    TEST_ASSERT_NEAR(r.y, 1.0f, EPS);
    TEST_ASSERT_NEAR(r.z, 0.0f, EPS);
    return true;
}

bool test_vec3_refract_perpendicular() {
    // Refract perpendicular ray (no bending)
    Vec3 v(0.0f, -1.0f, 0.0f);
    Vec3 n(0.0f, 1.0f, 0.0f);
    Vec3 r = refract(v, n, 1.0f);  // Same medium
    TEST_ASSERT_NEAR(r.x, 0.0f, EPS);
    TEST_ASSERT_NEAR(r.y, -1.0f, EPS);
    TEST_ASSERT_NEAR(r.z, 0.0f, EPS);
    return true;
}

bool test_vec3_unit_vector() {
    Vec3 v(3.0f, 4.0f, 0.0f);
    Vec3 u = unit_vector(v);
    TEST_ASSERT_NEAR(u.length(), 1.0f, EPS);
    return true;
}

// ==============================================================================
// Type Alias Tests
// ==============================================================================

bool test_point3_alias() {
    Point3 p(1.0f, 2.0f, 3.0f);
    TEST_ASSERT_NEAR(p.x, 1.0f, EPS);
    TEST_ASSERT_NEAR(p.y, 2.0f, EPS);
    TEST_ASSERT_NEAR(p.z, 3.0f, EPS);
    return true;
}

bool test_color_alias() {
    Color c(0.5f, 0.3f, 0.1f);
    TEST_ASSERT_NEAR(c.x, 0.5f, EPS);
    TEST_ASSERT_NEAR(c.y, 0.3f, EPS);
    TEST_ASSERT_NEAR(c.z, 0.1f, EPS);
    return true;
}

// ==============================================================================
// Test Suite Runner
// ==============================================================================

void run_vec3_tests(int& passed, int& failed, int& total) {
    // Construction
    RUN_TEST(test_vec3_default_constructor);
    RUN_TEST(test_vec3_value_constructor);
    RUN_TEST(test_vec3_single_value_constructor);

    // Operators
    RUN_TEST(test_vec3_negation);
    RUN_TEST(test_vec3_addition);
    RUN_TEST(test_vec3_subtraction);
    RUN_TEST(test_vec3_multiplication);
    RUN_TEST(test_vec3_scalar_multiplication);
    RUN_TEST(test_vec3_division);
    RUN_TEST(test_vec3_index_operator);
    RUN_TEST(test_vec3_compound_addition);
    RUN_TEST(test_vec3_compound_multiplication);
    RUN_TEST(test_vec3_compound_division);

    // Methods
    RUN_TEST(test_vec3_length);
    RUN_TEST(test_vec3_length_squared);
    RUN_TEST(test_vec3_normalized);
    RUN_TEST(test_vec3_near_zero);

    // Free functions
    RUN_TEST(test_vec3_dot);
    RUN_TEST(test_vec3_cross);
    RUN_TEST(test_vec3_cross_anticommutative);
    RUN_TEST(test_vec3_reflect);
    RUN_TEST(test_vec3_refract_perpendicular);
    RUN_TEST(test_vec3_unit_vector);

    // Type aliases
    RUN_TEST(test_point3_alias);
    RUN_TEST(test_color_alias);
}
