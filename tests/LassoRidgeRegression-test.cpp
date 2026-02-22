//
// Created by korone on 1/10/25.
//

#include "../src/Layers/Regularization/LassoRidgeRegression.h"
#include <gtest/gtest.h>

TEST(LassoRidgeRegressionTest, ComputeCombination) {
    // Suppose ridgeAlpha=1.0, lassoAlpha=2.0
    LassoRidgeRegression combo(1.0, 2.0);

    // Weights
    //   squares => (1^2 + 2^2 + 3^2 + (-4)^2) = 1 + 4 + 9 + 16 = 30
    //   abs     => (1   + 2   + 3   + 4     ) = 10
    //   total   => ridgeAlpha * 30 + lassoAlpha * 10
    //            = 1.0 * 30 + 2.0 * 10
    //            = 30 + 20 = 50
    std::vector<std::vector<double>> weights = {
        { 1.0,  2.0 },
        { 3.0, -4.0 }
    };
    Matrix weightsMatrix(weights);
    double result = combo.compute(weightsMatrix);
    EXPECT_DOUBLE_EQ(result, 50.0)
        << "LassoRidge with ridgeAlpha=1.0 and lassoAlpha=2.0 should produce 50.0.";
}

TEST(LassoRidgeRegressionTest, ComputeCombinationSmallerAlphas) {
    // ridgeAlpha=0.5, lassoAlpha=0.5
    LassoRidgeRegression combo(0.5, 0.5);

    // Weights
    // squares => 2^2 + (-1)^2 + 4^2 + 1^2 = 4 + 1 + 16 + 1 = 22
    // abs     => 2 + 1 + 4 + 1 = 8
    // total   => 0.5*22 + 0.5*8 = 11 + 4 = 15
    std::vector<std::vector<double>> weights = {
        {  2.0, -1.0 },
        {  4.0,  1.0 }
    };
    Matrix weightsMatrix(weights);

    double result = combo.compute(weightsMatrix);
    EXPECT_DOUBLE_EQ(result, 15.0)
        << "Should compute 15.0 with alpha=0.5 for both ridge and lasso.";
}

