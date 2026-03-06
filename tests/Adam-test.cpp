//
// Created by aapr6 on 1/10/25.
//

#include <gtest/gtest.h>
#include "../src/Optimizers/Adam/Adam.h"
#include "../src/Optimizers/Adam/AdamState.h"

namespace {
    constexpr double LEARNING_RATE = 0.1;
    constexpr double BETA1 = 0.9;
    constexpr double BETA2 = 0.999;
    constexpr double EPSILON = 1e-8;
    constexpr double TOLERANCE = 1e-5;

    // Shared test fixture data for bias/weight values and gradients
    const std::vector<double> INITIAL_VALUES = {
        -0.9705, -0.9054, 0.5125, -0.8614, 0.2941, 0.2524, -0.3645
    };
    const std::vector<double> GRADIENT_VALUES = {
        -0.4262, -0.4455, 0.1826, -0.2301, 0.4569, 0.4615, 0.6644
    };
    // Pre-computed expected values after one Adam update step
    const std::vector<double> EXPECTED_AFTER_UPDATE = {
        -0.8705000023, -0.8054000022, 0.4125000055, -0.7614000043,
        0.1941000022, 0.1524000022, -0.4644999985
    };
}

TEST(AdamTest, UpdateBias)
{
    uwu::Vector biasVector(INITIAL_VALUES);
    uwu::Vector gradientVector(GRADIENT_VALUES);

    Adam adam(LEARNING_RATE, BETA1, BETA2, EPSILON);
    AdamState state;
    adam.update(gradientVector, biasVector, state);

    for (size_t index = 0; index < biasVector.size(); ++index)
    {
        EXPECT_NEAR(biasVector[index], EXPECTED_AFTER_UPDATE[index], TOLERANCE);
    }
}

TEST(AdamTest, UpdateWeights)
{
    std::vector<std::vector<double>> weights = {INITIAL_VALUES, INITIAL_VALUES};
    Matrix weightsMatrix = Matrix(weights);

    std::vector<std::vector<double>> gradient = {GRADIENT_VALUES, GRADIENT_VALUES};
    Matrix gradientMatrix = Matrix(gradient);

    std::vector<std::vector<double>> expected = {EXPECTED_AFTER_UPDATE, EXPECTED_AFTER_UPDATE};

    Adam adam(LEARNING_RATE, BETA1, BETA2, EPSILON);
    AdamState state;
    adam.update(gradientMatrix, weightsMatrix, state);

    weightsMatrix.iterate([&](double& weight, int row, int col) {
        EXPECT_NEAR(weight, expected[row][col], TOLERANCE);
    });
}