//
// Created by aapr6 on 1/10/25.
//

#include <gtest/gtest.h>
#include "../src/Optimizers/RMSProp/RMSProp.h"
#include "../src/Optimizers/RMSProp/RMSPropState.h"

TEST(RMSPropTest, UpdateBias)
{
    std::vector<double> bias = {-0.9705, -0.9054, 0.5125, -0.8614, 0.2941, 0.2524, -0.3645};
    uwu::Vector biasVector = uwu::Vector(bias);

    std::vector<double> gradient = {-0.4262, -0.4455, 0.1826, -0.2301, 0.4569, 0.4615, 0.6644};
    uwu::Vector gradientVector = uwu::Vector(gradient);

    std::vector<double> expected = {-0.654272321, -0.5891723136, 0.1962727082, -0.5451725326, -0.02212769028, -0.06382769178, -0.6807277302};

    double alpha = 0.1;   // Learning rate
    double gamma = 0.9;   // Decay rate
    double epsilon = 1e-8;

    RMSProp rmsprop(alpha, gamma, epsilon);
    RMSPropState state;
    rmsprop.update(gradientVector, biasVector, state);

    // Comparar cada valor con tolerancia
    double tolerance = 1e-5;
    for (int i = 0; i < biasVector.size(); i++) {
        EXPECT_NEAR(biasVector[i], expected[i], tolerance);
    }
}

TEST(RMSPropTest, UpdateWeights)
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
        {-0.654272321, -0.5891723136, 0.1962727082, -0.5451725326, -0.02212769028, -0.06382769178, -0.6807277302},
        {-0.654272321, -0.5891723136, 0.1962727082, -0.5451725326, -0.02212769028, -0.06382769178, -0.6807277302}
    };

    double alpha = 0.1;   // Learning rate
    double gamma = 0.9;   // Decay rate
    double epsilon = 1e-8;

    RMSProp rmsprop(alpha, gamma, epsilon);
    RMSPropState state;
    rmsprop.update(gradientMatrix, weightsMatrix, state);

    // Comparar cada valor con tolerancia
    double tolerance = 1e-5;
    weightsMatrix.iterate([&](double& weight, int row, int col) {
        EXPECT_NEAR(weight, expected[row][col], tolerance);
    });
}