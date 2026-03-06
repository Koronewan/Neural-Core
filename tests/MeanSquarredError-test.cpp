//
// Created by korone on 1/10/25.
//

#include <gtest/gtest.h>
#include <vector>
#include "./Loss/MeanSquarredError.h"

namespace {
    constexpr double STRICT_TOLERANCE = 1e-12;
}

// Gradient should be zero when predictions exactly match expected values
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

// Gradient formula: 2 * (expected - item) / n
TEST(MeanSquaredErrorTest, GradientSimpleValues) {
    MeanSquarredError mse;

    // item={1,2,3}, expected={3,2,1}
    // diff = (3-1, 2-2, 1-3) = (2, 0, -2)
    // gradient = 2*diff / n = (4/3, 0, -4/3)
    std::vector<double> item = {1.0, 2.0, 3.0};
    std::vector<double> expected = {3.0, 2.0, 1.0};
    constexpr int vectorSize = 3;

    uwu::Vector grad = mse.gradient(item, expected);

    ASSERT_EQ(grad.size(), static_cast<size_t>(vectorSize));
    EXPECT_NEAR(grad[0],  4.0 / vectorSize, STRICT_TOLERANCE);
    EXPECT_NEAR(grad[1],  0.0,              STRICT_TOLERANCE);
    EXPECT_NEAR(grad[2], -4.0 / vectorSize, STRICT_TOLERANCE);
}

// Verify gradient handles negative values correctly
TEST(MeanSquaredErrorTest, GradientWithNegatives) {
    MeanSquarredError mse;

    // item={-1,-2}, expected={1,2}
    // diff = (1-(-1), 2-(-2)) = (2, 4)
    // gradient = 2*diff / n = (4/2, 8/2) = (2, 4)
    std::vector<double> item = {-1.0, -2.0};
    std::vector<double> expected = {1.0, 2.0};

    uwu::Vector grad = mse.gradient(item, expected);

    ASSERT_EQ(grad.size(), 2u);
    EXPECT_DOUBLE_EQ(grad[0], 2.0);
    EXPECT_DOUBLE_EQ(grad[1], 4.0);
}
