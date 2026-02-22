//
// Created by korone on 1/10/25.
//

#include "../src/Layers/Regularization/RidgeRegression.h"
#include <gtest/gtest.h>

TEST(RidgeRegressionTest, ComputeL2Regularization) {
    // Suppose alpha = 1.0
    RidgeRegression ridge(1.0);

    // Example weights
    // sum of squares => (1^2 + 2^2 + 3^2 + 4^2) = 1 + 4 + 9 + 16 = 30
    // multiplied by alpha=1 => 30
    std::vector<std::vector<double>> weights = {
        { 1.0,  2.0 },
        { 3.0,  4.0 }
    };
    Matrix weightsMatrix(weights);
    double result = ridge.compute(weightsMatrix);
    EXPECT_DOUBLE_EQ(result, 30.0)
        << "Ridge should compute sum of squares * alpha.";
}

TEST(RidgeRegressionTest, ComputeL2WithAlphaQuarter) {
    // alpha = 0.25
    RidgeRegression ridge(0.25);

    // sum of squares => (-1^2 + 2^2 + 3^2 + 4^2) = 1 + 4 + 9 + 16 = 30
    // times alpha=0.25 => 7.5
    std::vector<std::vector<double>> weights = {
        { -1.0, 2.0 },
        {  3.0, 4.0 }
    };
    Matrix weightsMatrix(weights);
    double result = ridge.compute(weightsMatrix);
    EXPECT_DOUBLE_EQ(result, 7.5)
        << "Ridge with alpha=0.25 should produce 7.5 for these weights.";
}
