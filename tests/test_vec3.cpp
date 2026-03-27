/**
 * @file test_vec3.cpp
 * @brief Tests Vec3 : ctors, ops, dot/cross, reflect/refract, alias
 */

#include <gtest/gtest.h>
#include <cmath>

#include "raytracer/core/vec3.cuh"

using namespace rt;

constexpr float EPS = 1e-6f;


/** @brief Ctor par defaut -> x,y,z = 0 */
TEST(Vec3Test, DefaultConstructor) {
    Vec3 v;
    EXPECT_NEAR(v.x, 0.0f, EPS);
    EXPECT_NEAR(v.y, 0.0f, EPS);
    EXPECT_NEAR(v.z, 0.0f, EPS);
}

/** @brief Ctor(1,2,3) stocke chaque composante */
TEST(Vec3Test, ValueConstructor) {
    Vec3 v(1.0f, 2.0f, 3.0f);
    EXPECT_NEAR(v.x, 1.0f, EPS);
    EXPECT_NEAR(v.y, 2.0f, EPS);
    EXPECT_NEAR(v.z, 3.0f, EPS);
}

/** @brief Ctor(5) -> vecteur uniforme (5,5,5) */
TEST(Vec3Test, SingleValueConstructor) {
    Vec3 v(5.0f);
    EXPECT_NEAR(v.x, 5.0f, EPS);
    EXPECT_NEAR(v.y, 5.0f, EPS);
    EXPECT_NEAR(v.z, 5.0f, EPS);
}


/** @brief -v inverse le signe de chaque composante */
TEST(Vec3Test, Negation) {
    Vec3 v(1.0f, -2.0f, 3.0f);
    Vec3 neg = -v;
    EXPECT_NEAR(neg.x, -1.0f, EPS);
    EXPECT_NEAR(neg.y, 2.0f, EPS);
    EXPECT_NEAR(neg.z, -3.0f, EPS);
}

/** @brief op+ composante par composante */
TEST(Vec3Test, Addition) {
    Vec3 a(1.0f, 2.0f, 3.0f);
    Vec3 b(4.0f, 5.0f, 6.0f);
    Vec3 c = a + b;
    EXPECT_NEAR(c.x, 5.0f, EPS);
    EXPECT_NEAR(c.y, 7.0f, EPS);
    EXPECT_NEAR(c.z, 9.0f, EPS);
}

/** @brief op- composante par composante */
TEST(Vec3Test, Subtraction) {
    Vec3 a(4.0f, 5.0f, 6.0f);
    Vec3 b(1.0f, 2.0f, 3.0f);
    Vec3 c = a - b;
    EXPECT_NEAR(c.x, 3.0f, EPS);
    EXPECT_NEAR(c.y, 3.0f, EPS);
    EXPECT_NEAR(c.z, 3.0f, EPS);
}

/** @brief op* vec*vec -> produit Hadamard */
TEST(Vec3Test, Multiplication) {
    Vec3 a(1.0f, 2.0f, 3.0f);
    Vec3 b(2.0f, 3.0f, 4.0f);
    Vec3 c = a * b;
    EXPECT_NEAR(c.x, 2.0f, EPS);
    EXPECT_NEAR(c.y, 6.0f, EPS);
    EXPECT_NEAR(c.z, 12.0f, EPS);
}

/** @brief v*2 == 2*v commutativite scalaire */
TEST(Vec3Test, ScalarMultiplication) {
    Vec3 v(1.0f, 2.0f, 3.0f);
    Vec3 c1 = v * 2.0f;
    Vec3 c2 = 2.0f * v;
    EXPECT_NEAR(c1.x, 2.0f, EPS);
    EXPECT_NEAR(c1.y, 4.0f, EPS);
    EXPECT_NEAR(c1.z, 6.0f, EPS);
    EXPECT_NEAR(c2.x, 2.0f, EPS);
    EXPECT_NEAR(c2.y, 4.0f, EPS);
    EXPECT_NEAR(c2.z, 6.0f, EPS);
}

/** @brief v/2 divise chaque composante */
TEST(Vec3Test, Division) {
    Vec3 v(2.0f, 4.0f, 6.0f);
    Vec3 c = v / 2.0f;
    EXPECT_NEAR(c.x, 1.0f, EPS);
    EXPECT_NEAR(c.y, 2.0f, EPS);
    EXPECT_NEAR(c.z, 3.0f, EPS);
}

/** @brief v[0]=x, v[1]=y, v[2]=z */
TEST(Vec3Test, IndexOperator) {
    Vec3 v(1.0f, 2.0f, 3.0f);
    EXPECT_NEAR(v[0], 1.0f, EPS);
    EXPECT_NEAR(v[1], 2.0f, EPS);
    EXPECT_NEAR(v[2], 3.0f, EPS);
}

/** @brief op+= modification en place */
TEST(Vec3Test, CompoundAddition) {
    Vec3 v(1.0f, 2.0f, 3.0f);
    v += Vec3(1.0f, 1.0f, 1.0f);
    EXPECT_NEAR(v.x, 2.0f, EPS);
    EXPECT_NEAR(v.y, 3.0f, EPS);
    EXPECT_NEAR(v.z, 4.0f, EPS);
}

/** @brief op*= scalaire en place */
TEST(Vec3Test, CompoundMultiplication) {
    Vec3 v(1.0f, 2.0f, 3.0f);
    v *= 2.0f;
    EXPECT_NEAR(v.x, 2.0f, EPS);
    EXPECT_NEAR(v.y, 4.0f, EPS);
    EXPECT_NEAR(v.z, 6.0f, EPS);
}

/** @brief op/= scalaire en place */
TEST(Vec3Test, CompoundDivision) {
    Vec3 v(2.0f, 4.0f, 6.0f);
    v /= 2.0f;
    EXPECT_NEAR(v.x, 1.0f, EPS);
    EXPECT_NEAR(v.y, 2.0f, EPS);
    EXPECT_NEAR(v.z, 3.0f, EPS);
}


/** @brief length() sur triangle 3-4-5 classique */
TEST(Vec3Test, Length) {
    Vec3 v(3.0f, 4.0f, 0.0f);
    EXPECT_NEAR(v.length(), 5.0f, EPS);
}

/** @brief length_squared() = 25, pas de sqrt */
TEST(Vec3Test, LengthSquared) {
    Vec3 v(3.0f, 4.0f, 0.0f);
    EXPECT_NEAR(v.length_squared(), 25.0f, EPS);
}

/** @brief normalized() -> norme=1, composantes 0.6/0.8/0 */
TEST(Vec3Test, Normalized) {
    Vec3 v(3.0f, 4.0f, 0.0f);
    Vec3 n = v.normalized();
    EXPECT_NEAR(n.length(), 1.0f, EPS);
    EXPECT_NEAR(n.x, 0.6f, EPS);
    EXPECT_NEAR(n.y, 0.8f, EPS);
    EXPECT_NEAR(n.z, 0.0f, EPS);
}

/** @brief near_zero() : ~1e-9 = true, (1,0,0) = false */
TEST(Vec3Test, NearZero) {
    Vec3 small(1e-9f, 1e-9f, 1e-9f);
    Vec3 large(1.0f, 0.0f, 0.0f);
    EXPECT_TRUE(small.near_zero() == true);
    EXPECT_TRUE(large.near_zero() == false);
}


/** @brief dot((1,2,3),(4,5,6)) = 32 */
TEST(Vec3Test, Dot) {
    Vec3 a(1.0f, 2.0f, 3.0f);
    Vec3 b(4.0f, 5.0f, 6.0f);
    float d = dot(a, b);
    EXPECT_NEAR(d, 32.0f, EPS);
}

/** @brief cross(x,y) = z, regle main droite */
TEST(Vec3Test, Cross) {
    Vec3 a(1.0f, 0.0f, 0.0f);
    Vec3 b(0.0f, 1.0f, 0.0f);
    Vec3 c = cross(a, b);
    EXPECT_NEAR(c.x, 0.0f, EPS);
    EXPECT_NEAR(c.y, 0.0f, EPS);
    EXPECT_NEAR(c.z, 1.0f, EPS);
}

/** @brief cross(a,b) = -cross(b,a) anticommutativite */
TEST(Vec3Test, CrossAnticommutative) {
    Vec3 a(1.0f, 0.0f, 0.0f);
    Vec3 b(0.0f, 1.0f, 0.0f);
    Vec3 c1 = cross(a, b);
    Vec3 c2 = cross(b, a);
    EXPECT_NEAR(c1.x, -c2.x, EPS);
    EXPECT_NEAR(c1.y, -c2.y, EPS);
    EXPECT_NEAR(c1.z, -c2.z, EPS);
}

/** @brief reflect (1,-1,0) sur n=(0,1,0) -> (1,1,0) */
TEST(Vec3Test, Reflect) {
    Vec3 v(1.0f, -1.0f, 0.0f);
    Vec3 n(0.0f, 1.0f, 0.0f);
    Vec3 r = reflect(v, n);
    EXPECT_NEAR(r.x, 1.0f, EPS);
    EXPECT_NEAR(r.y, 1.0f, EPS);
    EXPECT_NEAR(r.z, 0.0f, EPS);
}

/** @brief refract perp, eta=1 -> meme direction (Snell trivial) */
TEST(Vec3Test, RefractPerpendicular) {
    Vec3 v(0.0f, -1.0f, 0.0f);
    Vec3 n(0.0f, 1.0f, 0.0f);
    Vec3 r = refract(v, n, 1.0f);
    EXPECT_NEAR(r.x, 0.0f, EPS);
    EXPECT_NEAR(r.y, -1.0f, EPS);
    EXPECT_NEAR(r.z, 0.0f, EPS);
}

/** @brief unit_vector() -> norme = 1 */
TEST(Vec3Test, UnitVector) {
    Vec3 v(3.0f, 4.0f, 0.0f);
    Vec3 u = unit_vector(v);
    EXPECT_NEAR(u.length(), 1.0f, EPS);
}


/** @brief Point3 alias -> meme comportement que Vec3 */
TEST(Vec3Test, Point3Alias) {
    Point3 p(1.0f, 2.0f, 3.0f);
    EXPECT_NEAR(p.x, 1.0f, EPS);
    EXPECT_NEAR(p.y, 2.0f, EPS);
    EXPECT_NEAR(p.z, 3.0f, EPS);
}

/** @brief Color alias RGB -> fonctionne comme Vec3 */
TEST(Vec3Test, ColorAlias) {
    Color c(0.5f, 0.3f, 0.1f);
    EXPECT_NEAR(c.x, 0.5f, EPS);
    EXPECT_NEAR(c.y, 0.3f, EPS);
    EXPECT_NEAR(c.z, 0.1f, EPS);
}
