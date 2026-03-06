//
// Created by korone on 1/10/25.
//

#include "../src/Layers/Regularization/LassoRegression.h"
#include <gtest/gtest.h>

namespace {
    constexpr double ALPHA_FULL = 1.0;
    constexpr double ALPHA_HALF = 0.5;
}

TEST(LassoRegressionTest, ComputeL1Regularization) {
    LassoRegression lasso(ALPHA_FULL);

    // Weights: |1| + |-2| + |3| + |4| = 10
    // L1 penalty = alpha * sum_of_abs = 1.0 * 10 = 10.0
    std::vector<std::vector<double>> weights = {
        { 1.0, -2.0 },
        { 3.0,  4.0 }
    };
    Matrix weightsMatrix(weights);

    constexpr double expectedPenalty = 10.0;
    double result = lasso.compute(weightsMatrix);
    EXPECT_DOUBLE_EQ(result, expectedPenalty)
        << "Lasso should compute sum of absolute values * alpha.";
}

TEST(LassoRegressionTest, ComputeL1WithAlphaHalf) {
    LassoRegression lasso(ALPHA_HALF);

    // Weights: |1| + |-0.5| + |2| + |3| = 6.5
    // L1 penalty = alpha * sum_of_abs = 0.5 * 6.5 = 3.25
    std::vector<std::vector<double>> weights = {
        { 1.0, -0.5 },
        { 2.0,  3.0 }
    };
    Matrix weightsMatrix(weights);

    constexpr double expectedPenalty = 3.25;
    double result = lasso.compute(weightsMatrix);
    EXPECT_DOUBLE_EQ(result, expectedPenalty)
        << "Lasso with alpha=0.5 should be 3.25 for these weights.";
}
