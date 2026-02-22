//
// Created by korone on 1/8/25.
//

#include <gtest/gtest.h>
#include "../src/Layers/Activations/ReLU.h"

TEST(ReLUTest, ActivateFuncion)
{
    std::vector vectorInput = {-0.9705, -0.9054, 0.5125, -0.8614, 0.2941, 0.2524, -0.3645};
    uwu::Vector input(vectorInput);
    std::vector<double> expected = {0, 0, 0.5125, 0, 0.2941, 0.2524, 0};
    uwu::Vector expectedOutput = uwu::Vector(expected);

    ReLU relu;

    relu.activate(input);

    for (size_t i = 0; i < input.size(); ++i) {
        EXPECT_NEAR(input[i], expectedOutput[i], 1e-5);
    }
}

TEST(ReLUTest, DerivativeFuncion)
{
    std::vector vectorInput = {-0.9705, -0.9054, 0.5125, -0.8614, 0.2941, 0.2524, -0.3645};
    uwu::Vector input = uwu::Vector(vectorInput);
    std::vector<double> expected = {0, 0, 1, 0, 1, 1, 0};
    uwu::Vector expectedOutput = uwu::Vector(expected);

    ReLU relu;
    relu.derivative(input);

    EXPECT_EQ(input, expectedOutput);
}
