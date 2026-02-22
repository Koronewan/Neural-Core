//
// Created by korone on 1/8/25.
//

#include <gtest/gtest.h>
#include "../src/Optimizers/SGD/SGD.h"
#include "../src/Optimizers/SGD/SGDState.h"

TEST(SGDTest, UpdateBias)
{
    std::vector bias = {-0.9705, -0.9054, 0.5125, -0.8614, 0.2941, 0.2524, -0.3645};
    uwu::Vector biasVector = uwu::Vector(bias);

    std::vector gradient = {-0.4262, -0.4455, 0.1826, -0.2301, 0.4569, 0.4615, 0.6644};
    uwu::Vector gradientVector = uwu::Vector(gradient);

    std::vector expected = {-0.92788, -0.86085, 0.49424, -0.83839, 0.24841, 0.20625, -0.43094};

    SGD sgd(0.1);
    SGDState state;
    sgd.update(gradientVector, biasVector, state);

    // Comparar cada valor con tolerancia
    float tolerance = 1e-4; // Puedes ajustar la tolerancia según lo necesites
    for (int i = 0; i < biasVector.size(); i++) {
        EXPECT_NEAR(expected[i], biasVector[i], tolerance);
    }
}

TEST(SGDTest, UpdateWeights)
{
    std::vector<std::vector<double>> weights = {
        {-0.9705, -0.9054, 0.5125, -0.8614, 0.2941, 0.2524, -0.3645},
        {-0.9705, -0.9054, 0.5125, -0.8614, 0.2941, 0.2524, -0.3645}
    };
    Matrix weightsMatrix = Matrix(weights);

    std::vector<std::vector<double>> gradient = {
        {-0.4262, -0.4455, 0.1826, -0.2301, 0.4569, 0.4615, 0.6644},
        {-0.4262, -0.4455, 0.1826, -0.2301, 0.4569, 0.4615, 0.6644}
    };
    Matrix gradientMatrix = Matrix(gradient);

    std::vector<std::vector<double>> expected = {
        {-0.92788, -0.86085, 0.49424, -0.83839, 0.24841, 0.20625, -0.43094},
        {-0.92788, -0.86085, 0.49424, -0.83839, 0.24841, 0.20625, -0.43094}
    };

    SGD sgd(0.1);
    SGDState state;
    sgd.update(gradientMatrix, weightsMatrix, state);

    // Comparar cada valor con tolerancia
    double tolerance = 1e-4; // Puedes ajustar la tolerancia según lo necesites
    weightsMatrix.iterate([&](double& weight, int row, int col) {
        EXPECT_NEAR(weight, expected[row][col], tolerance);
    });
}

