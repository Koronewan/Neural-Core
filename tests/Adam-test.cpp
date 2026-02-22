//
// Created by aapr6 on 1/10/25.
//

#include <gtest/gtest.h>
#include "../src/Optimizers/Adam/Adam.h"
#include "../src/Optimizers/Adam/AdamState.h"

TEST(AdamTest, UpdateBias)
{
    std::vector<double> bias = {-0.9705, -0.9054, 0.5125, -0.8614, 0.2941, 0.2524, -0.3645};
    uwu::Vector biasVector(bias);
    std::vector<double> gradient = {-0.4262, -0.4455, 0.1826, -0.2301, 0.4569, 0.4615, 0.6644};
    uwu::Vector gradientVector(gradient);
    std::vector<double> expected = {-0.8705000023, -0.8054000022, 0.4125000055, -0.7614000043, 0.1941000022,
        0.1524000022, -0.4644999985};

    double learningRate = 0.1;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;

    Adam adam(learningRate, beta1, beta2, epsilon);
    AdamState state;
    adam.update(gradientVector, biasVector, state);

    // Comparar cada valor con tolerancia
    double tolerance = 1e-5;
    for (size_t index = 0; index < biasVector.size(); ++index)
    {
        EXPECT_NEAR(biasVector[index], expected[index], tolerance);
    }
}

TEST(AdamTest, UpdateWeights)
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
        {-0.8705000023, -0.8054000022, 0.4125000055, -0.7614000043, 0.1941000022,
        0.1524000022, -0.4644999985},
        {-0.8705000023, -0.8054000022, 0.4125000055, -0.7614000043, 0.1941000022,
            0.1524000022, -0.4644999985}
    };

    double learningRate = 0.1;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;

    Adam adam(learningRate, beta1, beta2, epsilon);
    AdamState state;

    adam.update(gradientMatrix, weightsMatrix, state);

    // Comparar cada valor con tolerancia
    double tolerance = 1e-5;
    weightsMatrix.iterate([&](double& weight, int row, int col) {
        EXPECT_NEAR(weight, expected[row][col], tolerance);
    });
}