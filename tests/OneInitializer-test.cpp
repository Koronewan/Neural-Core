//
// Created by korone on 1/9/25.
//

#include <gtest/gtest.h>
#include "Layers/Initializers/OneInitializer.h"


TEST(OneInitializerTest, InitializeToOnes) {
    // Set up dimensions (e.g., 3 outputs x 2 inputs)
    const int numOutputs = 3;
    const int numInputs = 2;

    // Prepare weights and bias containers
    std::vector<std::vector<double>> weights(numOutputs, std::vector<double>(numInputs));
    Matrix weightsMatrix(weights);

    std::vector<double> biases(numOutputs);
    uwu::Vector biasesVector(biases);

    // Create an instance of OneInitializer
    OneInitializer oneInit;
    oneInit.initialize(weightsMatrix, biasesVector);

    // Check that all biases are exactly 1.0
    for (int i = 0; i < biasesVector.size(); ++i) {
        EXPECT_EQ(biasesVector[i], 1.0);
    }

    // Check that all weights are exactly 1.0
    weightsMatrix.iterate([](double& w, int /*row*/, int /*col*/) {
        EXPECT_DOUBLE_EQ(w, 1.0) << "Weight should be initialized to 1.0";
    });
}
