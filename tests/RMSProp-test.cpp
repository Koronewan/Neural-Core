//
// Created by aapr6 on 1/10/25.
//

#include <gtest/gtest.h>
#include "../src/Optimizers/RMSProp/RMSProp.h"
#include "../src/Optimizers/RMSProp/RMSPropState.h"

namespace {
    constexpr double LEARNING_RATE = 0.1;
    constexpr double DECAY_RATE = 0.9;
    constexpr double EPSILON = 1e-8;
    constexpr double TOLERANCE = 1e-5;

    // Shared test fixture data
    const std::vector<double> INITIAL_VALUES = {
        -0.9705, -0.9054, 0.5125, -0.8614, 0.2941, 0.2524, -0.3645
    };
    const std::vector<double> GRADIENT_VALUES = {
        -0.4262, -0.4455, 0.1826, -0.2301, 0.4569, 0.4615, 0.6644
    };
    // Pre-computed expected values after one RMSProp update step
    const std::vector<double> EXPECTED_AFTER_UPDATE = {
        -0.654272321, -0.5891723136, 0.1962727082, -0.5451725326,
        -0.02212769028, -0.06382769178, -0.6807277302
    };
}

TEST(RMSPropTest, UpdateBias)
{
    uwu::Vector biasVector = uwu::Vector(INITIAL_VALUES);
    uwu::Vector gradientVector = uwu::Vector(GRADIENT_VALUES);

    RMSProp rmsprop(LEARNING_RATE, DECAY_RATE, EPSILON);
    RMSPropState state;
    rmsprop.update(gradientVector, biasVector, state);

    for (int i = 0; i < biasVector.size(); i++) {
        EXPECT_NEAR(biasVector[i], EXPECTED_AFTER_UPDATE[i], TOLERANCE);
    }
}

TEST(RMSPropTest, UpdateWeights)
{
    std::vector<std::vector<double>> weights = {INITIAL_VALUES, INITIAL_VALUES};
    Matrix weightsMatrix = Matrix(weights);

    std::vector<std::vector<double>> gradient = {GRADIENT_VALUES, GRADIENT_VALUES};
    Matrix gradientMatrix = Matrix(gradient);

    std::vector<std::vector<double>> expected = {EXPECTED_AFTER_UPDATE, EXPECTED_AFTER_UPDATE};

    RMSProp rmsprop(LEARNING_RATE, DECAY_RATE, EPSILON);
    RMSPropState state;
    rmsprop.update(gradientMatrix, weightsMatrix, state);

    weightsMatrix.iterate([&](double& weight, int row, int col) {
        EXPECT_NEAR(weight, expected[row][col], TOLERANCE);
    });
}