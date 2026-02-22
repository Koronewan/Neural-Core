//
// Created by korone on 1/10/25.
//

#include "../src/Layers/Regularization/LassoRegression.h"
#include <gtest/gtest.h>

TEST(LassoRegressionTest, ComputeL1Regularization) {
    // Suppose alpha = 1.0
    LassoRegression lasso(1.0);

    // Example weights
    // sum of abs = 1 + 2 + 3 + 4 = 10
    // multiplied by alpha=1 => 10
    std::vector<std::vector<double>> weights = {
        { 1.0, -2.0 },
        { 3.0,  4.0 }
    };
    Matrix weightsMatrix(weights);

    double result = lasso.compute(weightsMatrix);
    EXPECT_DOUBLE_EQ(result, 10.0)
        << "Lasso should compute sum of absolute values * alpha.";
}

TEST(LassoRegressionTest, ComputeL1WithAlphaHalf) {
    // Now alpha = 0.5
    LassoRegression lasso(0.5);

    // sum of abs => 1 + 0.5 + 2 + 3 = 6.5
    std::vector<std::vector<double>> weights = {
        { 1.0, -0.5 },
        { 2.0,  3.0 }
    };
    Matrix weightsMatrix(weights);

    // sum of abs = (1.0 + 0.5 + 2.0 + 3.0) = 6.5
    // times alpha(0.5) => 3.25
    double result = lasso.compute(weightsMatrix);
    EXPECT_DOUBLE_EQ(result, 3.25)
        << "Lasso with alpha=0.5 should be 3.25 for these weights.";
}
