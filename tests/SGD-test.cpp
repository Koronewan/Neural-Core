//
// Created by korone on 1/8/25.
//

#include <gtest/gtest.h>
#include "../src/Optimizers/SGD/SGD.h"
#include "../src/Optimizers/SGD/SGDState.h"

namespace {
    constexpr double LEARNING_RATE = 0.1;
    constexpr double TOLERANCE = 1e-4;

    // Shared test fixture data
    const std::vector<double> INITIAL_VALUES = {
        -0.9705, -0.9054, 0.5125, -0.8614, 0.2941, 0.2524, -0.3645
    };
    const std::vector<double> GRADIENT_VALUES = {
        -0.4262, -0.4455, 0.1826, -0.2301, 0.4569, 0.4615, 0.6644
    };
    // Expected: initial - learningRate * gradient
    const std::vector<double> EXPECTED_AFTER_UPDATE = {
        -0.92788, -0.86085, 0.49424, -0.83839, 0.24841, 0.20625, -0.43094
    };
}

TEST(SGDTest, UpdateBias)
{
    uwu::Vector biasVector = uwu::Vector(INITIAL_VALUES);
    uwu::Vector gradientVector = uwu::Vector(GRADIENT_VALUES);

    SGD sgd(LEARNING_RATE);
    SGDState state;
    sgd.update(gradientVector, biasVector, state);

    for (int i = 0; i < biasVector.size(); i++) {
        EXPECT_NEAR(EXPECTED_AFTER_UPDATE[i], biasVector[i], TOLERANCE);
    }
}

TEST(SGDTest, UpdateWeights)
{
    std::vector<std::vector<double>> weights = {INITIAL_VALUES, INITIAL_VALUES};
    Matrix weightsMatrix = Matrix(weights);

    std::vector<std::vector<double>> gradient = {GRADIENT_VALUES, GRADIENT_VALUES};
    Matrix gradientMatrix = Matrix(gradient);

    std::vector<std::vector<double>> expected = {EXPECTED_AFTER_UPDATE, EXPECTED_AFTER_UPDATE};

    SGD sgd(LEARNING_RATE);
    SGDState state;
    sgd.update(gradientMatrix, weightsMatrix, state);

    weightsMatrix.iterate([&](double& weight, int row, int col) {
        EXPECT_NEAR(weight, expected[row][col], TOLERANCE);
    });
}

