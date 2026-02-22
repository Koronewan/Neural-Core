//
// Created by korone on 1/9/25.
//

#include <gtest/gtest.h>
#include <random>
#include <cmath>
#include "./Layers/Initializers/GlorotInitializer.h"

// Simple test to check the range of the Glorot initialization
TEST(GlorotInitializerTest, InitializeWeightsInCorrectRange) {
    // Suppose we define nInputs = 3, nOutputs = 4
    // so we have a weights matrix of shape (4 x 3)
    // and a bias vector of size 4
    int nInputs = 3;
    int nOutputs = 4;

    std::vector<std::vector<double>> weights(nOutputs, std::vector<double>(nInputs));
    Matrix weightsMatrix(weights);
    std::vector<double> biases(nOutputs);
    uwu::Vector biasesVector(biases);

    GlorotInitializer glorot;
    glorot.initialize(weightsMatrix, biasesVector);

    // 1) Check that biases are zero
    for (int i = 0; i < biasesVector.size(); i++) {
        EXPECT_DOUBLE_EQ(biasesVector[i], 0.0) << "Bias should be initialized to zero.";
    }

    // 2) Check the range of the weights
    // The formula for the uniform distribution range is:
    //   +/- sqrt(6) / sqrt(nInputs + nOutputs)
    double limit = std::sqrt(6.0) / std::sqrt(nInputs + nOutputs);
    weightsMatrix.iterate([&](double& w, int /*row*/, int /*col*/)
    {
        EXPECT_GE(w, -limit) << "Weight is below the expected lower bound.";
        EXPECT_LE(w,  limit) << "Weight is above the expected upper bound.";
    });

    // 3) Optional quick check: ensure not all weights are the same
    //    (this is a coarse check for randomness)
    // We just store the first weight:
    double firstWeight = weights[0][0];
    bool allSame = true;
    weightsMatrix.iterate([&](double& w, int /*row*/, int /*col*/)
        {
        if (std::fabs(w - firstWeight) > 1e-12)
        {
            allSame = false;
        }
    });
    EXPECT_FALSE(allSame) << "All weights are the same; expected some randomness.";
}
