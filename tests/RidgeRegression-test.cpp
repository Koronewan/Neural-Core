//
// Created by korone on 1/10/25.
//

#include "../src/Layers/Regularization/RidgeRegression.h"
#include <gtest/gtest.h>

namespace {
    constexpr double ALPHA_FULL = 1.0;
    constexpr double ALPHA_QUARTER = 0.25;
}

TEST(RidgeRegressionTest, ComputeL2Regularization) {
    RidgeRegression ridge(ALPHA_FULL);

    // Weights: 1^2 + 2^2 + 3^2 + 4^2 = 30
    // L2 penalty = alpha * sum_of_squares = 1.0 * 30 = 30.0
    std::vector<std::vector<double>> weights = {
        { 1.0,  2.0 },
        { 3.0,  4.0 }
    };
    Matrix weightsMatrix(weights);

    constexpr double expectedPenalty = 30.0;
    double result = ridge.compute(weightsMatrix);
    EXPECT_DOUBLE_EQ(result, expectedPenalty)
        << "Ridge should compute sum of squares * alpha.";
}

TEST(RidgeRegressionTest, ComputeL2WithAlphaQuarter) {
    RidgeRegression ridge(ALPHA_QUARTER);

    // Weights: (-1)^2 + 2^2 + 3^2 + 4^2 = 30
    // L2 penalty = alpha * sum_of_squares = 0.25 * 30 = 7.5
    std::vector<std::vector<double>> weights = {
        { -1.0, 2.0 },
        {  3.0, 4.0 }
    };
    Matrix weightsMatrix(weights);

    constexpr double expectedPenalty = 7.5;
    double result = ridge.compute(weightsMatrix);
    EXPECT_DOUBLE_EQ(result, expectedPenalty)
        << "Ridge with alpha=0.25 should produce 7.5 for these weights.";
}
