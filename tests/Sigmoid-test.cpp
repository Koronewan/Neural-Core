//
// Created by aapr6 on 1/11/25.
//

#include <gtest/gtest.h>
#include <cmath>
#include "../src/Layers/Activations/Sigmoid.h"
#include "../src/MathUtils/Vector.h"

TEST(SigmoidTest, ActivateFunction)
{
    std::vector vectorInput = {-0.9705, -0.9054, 0.5125, -0.8614, 0.2941, 0.2524, -0.3645};
    uwu::Vector input = uwu::Vector(vectorInput);
    std::vector<double> expected = {
        0.2747808528,
        0.2879420611,
        0.6253923497,
        0.2970469285,
        0.5729995832,
        0.5627671343,
        0.409870681
    };
    uwu::Vector expectedOutput = uwu::Vector(expected);

    Sigmoid sigmoid;

    sigmoid.activate(input);

    for (size_t i = 0; i < input.size(); ++i) {
        EXPECT_NEAR(input[i], expectedOutput[i], 1e-5);
    }
}

TEST(SigmoidTest, DerivativeFunction)
{
    std::vector<double> vectorInput = {-0.9705, -0.9054, 0.5125, -0.8614, 0.2941, 0.2524, -0.3645};
    uwu::Vector input = uwu::Vector(vectorInput);
    std::vector<double> expected;

    for (double x : vectorInput)
    {
        double sigmoidValue = 1.0 / (1.0 + std::exp(-x));
        expected.push_back(sigmoidValue * (1.0 - sigmoidValue));
    }

    uwu::Vector expectedOutput = uwu::Vector(expected);

    Sigmoid sigmoid;
    sigmoid.derivative(input);

    EXPECT_EQ(input, expectedOutput);
}