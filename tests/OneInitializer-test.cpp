//
// Created by korone on 1/9/25.
//

#include <gtest/gtest.h>
#include "Layers/Initializers/OneInitializer.h"

namespace {
    constexpr int NUM_OUTPUTS = 3;
    constexpr int NUM_INPUTS = 2;
    constexpr double EXPECTED_INIT_VALUE = 1.0;
    constexpr double EXPECTED_BIAS_INIT_VALUE = 0.0;
}

TEST(OneInitializerTest, InitializeToOnes) {
    std::vector<std::vector<double>> weights(NUM_OUTPUTS, std::vector<double>(NUM_INPUTS));
    Matrix weightsMatrix(weights);

    std::vector<double> biases(NUM_OUTPUTS);
    uwu::Vector biasesVector(biases);

    OneInitializer oneInit;
    oneInit.initialize(weightsMatrix, biasesVector);

    // All biases should be exactly 0.0
    for (int i = 0; i < biasesVector.size(); ++i) {
        EXPECT_EQ(biasesVector[i], EXPECTED_BIAS_INIT_VALUE);
    }

    // All weights should be exactly 1.0
    weightsMatrix.iterate([](double& w, int /*row*/, int /*col*/) {
        EXPECT_DOUBLE_EQ(w, EXPECTED_INIT_VALUE) << "Weight should be initialized to 1.0";
    });
}
