/**
 * @file test_interval.cpp
 * @brief Tests Interval : ctors, size, contains/surrounds, clamp, expand
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cfloat>

#include "raytracer/core/interval.cuh"

using namespace rt;

constexpr float EPS = 1e-6f;


/** @brief Ctor par defaut -> intervalle vide (min > max) */
TEST(IntervalTest, DefaultConstructor) {
    Interval i;
    EXPECT_TRUE(i.min > i.max);
}

/** @brief Ctor(1,5) stocke min=1, max=5 */
TEST(IntervalTest, ValueConstructor) {
    Interval i(1.0f, 5.0f);
    EXPECT_NEAR(i.min, 1.0f, EPS);
    EXPECT_NEAR(i.max, 5.0f, EPS);
}

/** @brief Fusion [1,3]+[2,5] -> [1,5] (chevauchement) */
TEST(IntervalTest, MergeConstructor) {
    Interval a(1.0f, 3.0f);
    Interval b(2.0f, 5.0f);
    Interval merged(a, b);
    EXPECT_NEAR(merged.min, 1.0f, EPS);
    EXPECT_NEAR(merged.max, 5.0f, EPS);
}

/** @brief Fusion [1,2]+[4,5] -> [1,5] (disjoints) */
TEST(IntervalTest, MergeDisjoint) {
    Interval a(1.0f, 2.0f);
    Interval b(4.0f, 5.0f);
    Interval merged(a, b);
    EXPECT_NEAR(merged.min, 1.0f, EPS);
    EXPECT_NEAR(merged.max, 5.0f, EPS);
}


/** @brief [2,7] : size() = 5 */
TEST(IntervalTest, Size) {
    Interval i(2.0f, 7.0f);
    EXPECT_NEAR(i.size(), 5.0f, EPS);
}

/** @brief [3,3] degenere -> size() = 0 */
TEST(IntervalTest, SizeZero) {
    Interval i(3.0f, 3.0f);
    EXPECT_NEAR(i.size(), 0.0f, EPS);
}

/** @brief [5,2] inverse -> size() < 0 (invalide) */
TEST(IntervalTest, SizeNegative) {
    Interval i(5.0f, 2.0f);
    EXPECT_TRUE(i.size() < 0.0f);
}


/** @brief 5 dans [0,10] -> contains = true */
TEST(IntervalTest, ContainsInside) {
    Interval i(0.0f, 10.0f);
    EXPECT_TRUE(i.contains(5.0f) == true);
}

/** @brief borne min incluse (ferme) -> contains(0) = true */
TEST(IntervalTest, ContainsAtMin) {
    Interval i(0.0f, 10.0f);
    EXPECT_TRUE(i.contains(0.0f) == true);
}

/** @brief borne max incluse (ferme) -> contains(10) = true */
TEST(IntervalTest, ContainsAtMax) {
    Interval i(0.0f, 10.0f);
    EXPECT_TRUE(i.contains(10.0f) == true);
}

/** @brief -1 sous [0,10] -> contains = false */
TEST(IntervalTest, ContainsOutsideBelow) {
    Interval i(0.0f, 10.0f);
    EXPECT_TRUE(i.contains(-1.0f) == false);
}

/** @brief 11 au dessus de [0,10] -> contains = false */
TEST(IntervalTest, ContainsOutsideAbove) {
    Interval i(0.0f, 10.0f);
    EXPECT_TRUE(i.contains(11.0f) == false);
}


/** @brief surrounds(5) dans ]0,10[ -> true (ouvert) */
TEST(IntervalTest, SurroundsInside) {
    Interval i(0.0f, 10.0f);
    EXPECT_TRUE(i.surrounds(5.0f) == true);
}

/** @brief surrounds(0) = false, borne exclue */
TEST(IntervalTest, SurroundsAtMin) {
    Interval i(0.0f, 10.0f);
    EXPECT_TRUE(i.surrounds(0.0f) == false);
}

/** @brief surrounds(10) = false, max exclue aussi */
TEST(IntervalTest, SurroundsAtMax) {
    Interval i(0.0f, 10.0f);
    EXPECT_TRUE(i.surrounds(10.0f) == false);
}

/** @brief surrounds(0.001) juste au dessus de min -> true */
TEST(IntervalTest, SurroundsNearMin) {
    Interval i(0.0f, 10.0f);
    EXPECT_TRUE(i.surrounds(0.001f) == true);
}


/** @brief clamp(5) dans [0,10] -> inchange */
TEST(IntervalTest, ClampInside) {
    Interval i(0.0f, 10.0f);
    EXPECT_NEAR(i.clamp(5.0f), 5.0f, EPS);
}

/** @brief clamp(-5) -> ramene a min=0 */
TEST(IntervalTest, ClampBelow) {
    Interval i(0.0f, 10.0f);
    EXPECT_NEAR(i.clamp(-5.0f), 0.0f, EPS);
}

/** @brief clamp(15) -> sature a max=10 */
TEST(IntervalTest, ClampAbove) {
    Interval i(0.0f, 10.0f);
    EXPECT_NEAR(i.clamp(15.0f), 10.0f, EPS);
}

/** @brief clamp aux bornes exactes -> pas de changement */
TEST(IntervalTest, ClampAtBoundary) {
    Interval i(0.0f, 10.0f);
    EXPECT_NEAR(i.clamp(0.0f), 0.0f, EPS);
    EXPECT_NEAR(i.clamp(10.0f), 10.0f, EPS);
}


/** @brief expand(2) sur [2,8] -> [1,9] (delta/2 par cote) */
TEST(IntervalTest, Expand) {
    Interval i(2.0f, 8.0f);
    Interval expanded = i.expand(2.0f);
    EXPECT_NEAR(expanded.min, 1.0f, EPS);
    EXPECT_NEAR(expanded.max, 9.0f, EPS);
}

/** @brief expand(0) -> noop */
TEST(IntervalTest, ExpandZero) {
    Interval i(2.0f, 8.0f);
    Interval expanded = i.expand(0.0f);
    EXPECT_NEAR(expanded.min, 2.0f, EPS);
    EXPECT_NEAR(expanded.max, 8.0f, EPS);
}

/** @brief expand(-4) retrecit [0,10] -> [2,8] */
TEST(IntervalTest, ExpandNegative) {
    Interval i(0.0f, 10.0f);
    Interval shrunk = i.expand(-4.0f);
    EXPECT_NEAR(shrunk.min, 2.0f, EPS);
    EXPECT_NEAR(shrunk.max, 8.0f, EPS);
}
