//
// Created by korone on 1/10/25.
//

#include <gtest/gtest.h>
#include <vector>
#include "./Loss/MeanSquarredError.h"

// Test 1: Check gradient when both vectors are identical (expect zeros).
TEST(MeanSquaredErrorTest, GradientIdenticalVectors) {
    MeanSquarredError mse;

    std::vector item = {0.0, 0.0, 0.0};
    std::vector expected = {0.0, 0.0, 0.0};

    uwu::Vector grad = mse.gradient(item, expected);

    // The difference (expected[i] - item[i]) is zero => gradient should be zero.
    ASSERT_EQ(grad.size(), item.size());
    for (size_t i = 0; i < grad.size(); ++i) {
        EXPECT_DOUBLE_EQ(grad[i], 0.0);
    }
}

// Test 2: Check gradient with simple numeric values
TEST(MeanSquaredErrorTest, GradientSimpleValues) {
    MeanSquarredError mse;

    // item = {1, 2, 3}, expected = {3, 2, 1}
    // difference: (3-1, 2-2, 1-3) = (2, 0, -2)
    // multiply by 2 => (4, 0, -4)
    // divide by size()=3 => (4/3, 0, -4/3)
    std::vector<double> item = {1.0, 2.0, 3.0};
    std::vector<double> expected = {3.0, 2.0, 1.0};

    uwu::Vector grad = mse.gradient(item, expected);

    // Check results:
    ASSERT_EQ(grad.size(), 3u);
    EXPECT_NEAR(grad[0],  4.0/3.0, 1e-12);
    EXPECT_NEAR(grad[1],  0.0,    1e-12);
    EXPECT_NEAR(grad[2], -4.0/3.0, 1e-12);
}

// Test 3: Check a non-trivial case with negative values
TEST(MeanSquaredErrorTest, GradientWithNegatives) {
    MeanSquarredError mse;

    // item = {-1, -2}, expected = {1, 2}
    // difference: (1 - (-1), 2 - (-2)) = (2, 4)
    // multiply by 2 => (4, 8)
    // divide by size()=2 => (2, 4)
    std::vector<double> item = {-1.0, -2.0};
    std::vector<double> expected = {1.0, 2.0};

    uwu::Vector grad = mse.gradient(item, expected);

    ASSERT_EQ(grad.size(), 2u);
    EXPECT_DOUBLE_EQ(grad[0], 2.0);
    EXPECT_DOUBLE_EQ(grad[1], 4.0);
}
