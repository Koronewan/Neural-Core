//
// Created by korone on 1/10/25.
//

#include "../src/Layers/Regularization/LassoRidgeRegression.h"
#include <gtest/gtest.h>

namespace {
    constexpr double RIDGE_ALPHA_FULL = 1.0;
    constexpr double LASSO_ALPHA_DOUBLE = 2.0;
    constexpr double ALPHA_HALF = 0.5;
}

TEST(LassoRidgeRegressionTest, ComputeCombination) {
    LassoRidgeRegression combo(RIDGE_ALPHA_FULL, LASSO_ALPHA_DOUBLE);

    // Weights: squares = 1^2 + 2^2 + 3^2 + (-4)^2 = 30
    //          abs     = 1 + 2 + 3 + 4 = 10
    // Total = ridgeAlpha * 30 + lassoAlpha * 10 = 1.0*30 + 2.0*10 = 50.0
    std::vector<std::vector<double>> weights = {
        { 1.0,  2.0 },
        { 3.0, -4.0 }
    };
    Matrix weightsMatrix(weights);

    constexpr double expectedPenalty = 50.0;
    double result = combo.compute(weightsMatrix);
    EXPECT_DOUBLE_EQ(result, expectedPenalty)
        << "LassoRidge with ridgeAlpha=1.0 and lassoAlpha=2.0 should produce 50.0.";
}

TEST(LassoRidgeRegressionTest, ComputeCombinationSmallerAlphas) {
    LassoRidgeRegression combo(ALPHA_HALF, ALPHA_HALF);

    // Weights: squares = 2^2 + (-1)^2 + 4^2 + 1^2 = 22
    //          abs     = 2 + 1 + 4 + 1 = 8
    // Total = 0.5 * 22 + 0.5 * 8 = 11 + 4 = 15.0
    std::vector<std::vector<double>> weights = {
        {  2.0, -1.0 },
        {  4.0,  1.0 }
    };
    Matrix weightsMatrix(weights);

    constexpr double expectedPenalty = 15.0;
    double result = combo.compute(weightsMatrix);
    EXPECT_DOUBLE_EQ(result, expectedPenalty)
        << "Should compute 15.0 with alpha=0.5 for both ridge and lasso.";
}

