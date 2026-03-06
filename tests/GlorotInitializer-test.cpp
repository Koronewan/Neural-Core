//
// Created by korone on 1/9/25.
//

#include <gtest/gtest.h>
#include <random>
#include <cmath>
#include "./Layers/Initializers/GlorotInitializer.h"

namespace {
    constexpr int NUM_INPUTS = 3;
    constexpr int NUM_OUTPUTS = 4;
    constexpr double FLOAT_COMPARISON_EPSILON = 1e-12;
}

// Verify Glorot uniform initialization produces weights within the theoretical bounds
TEST(GlorotInitializerTest, InitializeWeightsInCorrectRange) {
    std::vector<std::vector<double>> weights(NUM_OUTPUTS, std::vector<double>(NUM_INPUTS));
    Matrix weightsMatrix(weights);
    std::vector<double> biases(NUM_OUTPUTS);
    uwu::Vector biasesVector(biases);

    GlorotInitializer glorot;
    glorot.initialize(weightsMatrix, biasesVector);

    // Biases should be initialized to zero
    for (int i = 0; i < biasesVector.size(); i++) {
        EXPECT_DOUBLE_EQ(biasesVector[i], 0.0) << "Bias should be initialized to zero.";
    }

    // Weights should fall within +/- sqrt(6 / (nInputs + nOutputs))
    const double glorotLimit = std::sqrt(6.0) / std::sqrt(NUM_INPUTS + NUM_OUTPUTS);
    weightsMatrix.iterate([&](double& w, int /*row*/, int /*col*/)
    {
        EXPECT_GE(w, -glorotLimit) << "Weight is below the expected lower bound.";
        EXPECT_LE(w,  glorotLimit) << "Weight is above the expected upper bound.";
    });

    // Coarse randomness check: not all weights should be identical
    double firstWeight = weights[0][0];
    bool allSame = true;
    weightsMatrix.iterate([&](double& w, int /*row*/, int /*col*/)
        {
        if (std::fabs(w - firstWeight) > FLOAT_COMPARISON_EPSILON)
        {
            allSame = false;
        }
    });
    EXPECT_FALSE(allSame) << "All weights are the same; expected some randomness.";
}
