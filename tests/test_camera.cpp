/**
 * @file test_camera.cpp
 * @brief Tests Camera : init, defocus blur, get_ray, shutter, FOV
 */

#include <gtest/gtest.h>
#include <cmath>

#include "raytracer/core/vec3.cuh"
#include "raytracer/core/ray.cuh"
#include "raytracer/core/random.cuh"

#ifndef __CUDACC__
inline float curand_uniform(curandState*) { return 0.5f; }
namespace rt {
    inline Vec3 random_unit_vector(curandState*) { return Vec3(0, 1, 0); }
    inline Vec3 random_in_unit_sphere(curandState*) { return Vec3(0, 0, 0); }
    inline Vec3 random_in_unit_disk(curandState*) { return Vec3(0, 0, 0); }
}
#endif

#include "raytracer/camera/camera.cuh"

using namespace rt;


/** @brief Init camera -> center = origine */
TEST(CameraTest, Center) {
    Camera cam;
    cam.initialize(800, 600, Point3(0, 0, 0), Point3(0, 0, -1), Vec3(0, 1, 0), 90.0f);
    EXPECT_NEAR(cam.center.x, 0.0f, 1e-5f);
    EXPECT_NEAR(cam.center.y, 0.0f, 1e-5f);
    EXPECT_NEAR(cam.center.z, 0.0f, 1e-5f);
}

/** @brief Dimensions 800x600 stockees correctement */
TEST(CameraTest, Dimensions) {
    Camera cam;
    cam.initialize(800, 600, Point3(0, 0, 0), Point3(0, 0, -1), Vec3(0, 1, 0), 90.0f);
    EXPECT_TRUE(cam.image_width == 800);
    EXPECT_TRUE(cam.image_height == 600);
}

/** @brief Repere local : w=+Z, u=+X, v=+Y quand lookAt=-Z */
TEST(CameraTest, CoordinateSystem) {
    Camera cam;
    cam.initialize(800, 600, Point3(0, 0, 0), Point3(0, 0, -1), Vec3(0, 1, 0), 90.0f);
    EXPECT_NEAR(cam.w.z, 1.0f, 1e-5f);
    EXPECT_NEAR(cam.u.x, 1.0f, 1e-5f);
    EXPECT_NEAR(cam.v.y, 1.0f, 1e-5f);
}

/** @brief Pas de defocus -> angle = 0 */
TEST(CameraTest, NoDefocus) {
    Camera cam;
    cam.initialize(800, 600, Point3(0, 0, 0), Point3(0, 0, -1), Vec3(0, 1, 0), 90.0f, 0.0f);
    EXPECT_NEAR(cam.defocus_angle, 0.0f, 1e-5f);
}

/** @brief Defocus angle=2 -> disk_u/v non nuls */
TEST(CameraTest, WithDefocus) {
    Camera cam;
    cam.initialize(800, 600, Point3(0, 0, 0), Point3(0, 0, -1), Vec3(0, 1, 0), 90.0f, 2.0f, 10.0f);
    EXPECT_NEAR(cam.defocus_angle, 2.0f, 1e-5f);
    EXPECT_TRUE(cam.defocus_disk_u.length() > 0.0f);
    EXPECT_TRUE(cam.defocus_disk_v.length() > 0.0f);
}


/** @brief get_ray pixel central -> origine cam, direction vers -Z */
TEST(CameraTest, RayFromCenter) {
    Camera cam;
    cam.initialize(800, 600, Point3(0, 0, 0), Point3(0, 0, -1), Vec3(0, 1, 0), 90.0f);
    CPURandom rng(42);
    Ray r = cam.get_ray_cpu(400, 300, rng);
    EXPECT_NEAR(r.origin().x, 0.0f, 1e-5f);
    EXPECT_NEAR(r.origin().y, 0.0f, 1e-5f);
    EXPECT_NEAR(r.origin().z, 0.0f, 1e-5f);
    Vec3 dir = unit_vector(r.direction());
    EXPECT_TRUE(dir.z < -0.5f);
}

/** @brief Coins opposes (0,0) vs (799,599) -> directions divergentes */
TEST(CameraTest, RayCornerDiverges) {
    Camera cam;
    cam.initialize(800, 600, Point3(0, 0, 0), Point3(0, 0, -1), Vec3(0, 1, 0), 90.0f);
    CPURandom rng(42);
    Ray r1 = cam.get_ray_cpu(0, 0, rng);
    Ray r2 = cam.get_ray_cpu(799, 599, rng);
    Vec3 d1 = unit_vector(r1.direction());
    Vec3 d2 = unit_vector(r2.direction());
    float cosine = dot(d1, d2);
    EXPECT_TRUE(cosine < 0.9f);
}

/** @brief Shutter time du rayon dans [0, 1] */
TEST(CameraTest, ShutterTime) {
    Camera cam;
    cam.initialize(800, 600, Point3(0, 0, 0), Point3(0, 0, -1), Vec3(0, 1, 0), 90.0f,
                   0.0f, 10.0f, 0.0f, 1.0f);
    CPURandom rng(42);
    Ray r = cam.get_ray_cpu(400, 300, rng);
    EXPECT_TRUE(r.time() >= 0.0f);
    EXPECT_TRUE(r.time() <= 1.0f);
}

/** @brief FOV 90 -> pixel_delta plus grand que FOV 20 */
TEST(CameraTest, FovAffectsSpread) {
    Camera cam_narrow, cam_wide;
    cam_narrow.initialize(800, 600, Point3(0, 0, 0), Point3(0, 0, -1), Vec3(0, 1, 0), 20.0f);
    cam_wide.initialize(800, 600, Point3(0, 0, 0), Point3(0, 0, -1), Vec3(0, 1, 0), 90.0f);
    float narrow_delta = cam_narrow.pixel_delta_u.length();
    float wide_delta = cam_wide.pixel_delta_u.length();
    EXPECT_TRUE(wide_delta > narrow_delta);
}
