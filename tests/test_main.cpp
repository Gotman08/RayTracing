/**
 * Unit Tests - Main Entry Point
 * Lightweight test framework without external dependencies
 */

#include <iostream>
#include <cmath>
#include <string>

// ==============================================================================
// Test Framework Macros
// ==============================================================================

#define TEST_ASSERT(expr) \
    do { \
        if (!(expr)) { \
            std::cerr << "    FAILED: " << #expr << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            return false; \
        } \
    } while(0)

#define TEST_ASSERT_MSG(expr, msg) \
    do { \
        if (!(expr)) { \
            std::cerr << "    FAILED: " << msg << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            return false; \
        } \
    } while(0)

#define TEST_ASSERT_NEAR(a, b, eps) \
    do { \
        if (std::abs((a) - (b)) > (eps)) { \
            std::cerr << "    FAILED: " << #a << " (" << (a) << ") != " << #b << " (" << (b) << ")" \
                      << " (eps=" << (eps) << ") at " << __FILE__ << ":" << __LINE__ << "\n"; \
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

// ==============================================================================
// Test Suite Registry
// ==============================================================================

// Forward declarations from test files
void run_vec3_tests(int& passed, int& failed, int& total);
void run_ray_tests(int& passed, int& failed, int& total);
void run_interval_tests(int& passed, int& failed, int& total);
void run_sphere_tests(int& passed, int& failed, int& total);
void run_plane_tests(int& passed, int& failed, int& total);
void run_aabb_tests(int& passed, int& failed, int& total);
void run_materials_tests(int& passed, int& failed, int& total);
void run_camera_tests(int& passed, int& failed, int& total);
void run_tone_mapping_tests(int& passed, int& failed, int& total);
void run_bvh_tests(int& passed, int& failed, int& total);

// ==============================================================================
// Main
// ==============================================================================

int main() {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "   CUDA Ray Tracer - Unit Tests\n";
    std::cout << "========================================\n\n";

    int total_passed = 0;
    int total_failed = 0;
    int total_tests = 0;

    // Run all test suites
    std::cout << "[Vec3 Tests]\n";
    run_vec3_tests(total_passed, total_failed, total_tests);
    std::cout << "\n";

    std::cout << "[Ray Tests]\n";
    run_ray_tests(total_passed, total_failed, total_tests);
    std::cout << "\n";

    std::cout << "[Interval Tests]\n";
    run_interval_tests(total_passed, total_failed, total_tests);
    std::cout << "\n";

    std::cout << "[Sphere Tests]\n";
    run_sphere_tests(total_passed, total_failed, total_tests);
    std::cout << "\n";

    std::cout << "[Plane Tests]\n";
    run_plane_tests(total_passed, total_failed, total_tests);
    std::cout << "\n";

    std::cout << "[AABB Tests]\n";
    run_aabb_tests(total_passed, total_failed, total_tests);
    std::cout << "\n";

    std::cout << "[Materials Tests]\n";
    run_materials_tests(total_passed, total_failed, total_tests);
    std::cout << "\n";

    std::cout << "[Camera Tests]\n";
    run_camera_tests(total_passed, total_failed, total_tests);
    std::cout << "\n";

    std::cout << "[Tone Mapping Tests]\n";
    run_tone_mapping_tests(total_passed, total_failed, total_tests);
    std::cout << "\n";

    std::cout << "[BVH Tests]\n";
    run_bvh_tests(total_passed, total_failed, total_tests);
    std::cout << "\n";

    // Summary
    std::cout << "========================================\n";
    std::cout << "   Results: " << total_passed << "/" << total_tests << " passed";
    if (total_failed > 0) {
        std::cout << " (" << total_failed << " failed)";
    }
    std::cout << "\n";
    std::cout << "========================================\n\n";

    return (total_failed > 0) ? 1 : 0;
}
