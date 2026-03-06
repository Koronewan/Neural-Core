//
// Created by aapr6 on 1/11/25.
//

#include <gtest/gtest.h>
#include <cmath>
#include "../src/Layers/Activations/Sigmoid.h"
#include "../src/MathUtils/Vector.h"

namespace {
    constexpr double TOLERANCE = 1e-5;

    // Mix of negative and positive values to exercise the full sigmoid curve
    const std::vector<double> TEST_INPUT = {
        -0.9705, -0.9054, 0.5125, -0.8614, 0.2941, 0.2524, -0.3645
    };
}

TEST(SigmoidTest, ActivateFunction)
{
    uwu::Vector input = uwu::Vector(TEST_INPUT);

    // Sigmoid: σ(x) = 1 / (1 + e^(−x))
    std::vector<double> expected;
    for (double x : TEST_INPUT)
    {
        expected.push_back(1.0 / (1.0 + std::exp(-x)));
    }
    uwu::Vector expectedOutput = uwu::Vector(expected);

    Sigmoid sigmoid;
    sigmoid.activate(input);

    for (size_t i = 0; i < input.size(); ++i) 
    {
        EXPECT_NEAR(input[i], expectedOutput[i], TOLERANCE);
    }
}

TEST(SigmoidTest, DerivativeFunction)
{
    uwu::Vector input = uwu::Vector(TEST_INPUT);

    // Sigmoid derivative: σ(x) * (1 − σ(x))
    std::vector<double> expected;
    for (double x : TEST_INPUT)
    {
        double sigmoidValue = 1.0 / (1.0 + std::exp(-x));
        expected.push_back(sigmoidValue * (1.0 - sigmoidValue));
    }
    uwu::Vector expectedOutput = uwu::Vector(expected);

    Sigmoid sigmoid;
    sigmoid.derivative(input);

    for (size_t i = 0; i < input.size(); ++i) 
    {
        EXPECT_NEAR(input[i], expectedOutput[i], TOLERANCE);
    }
}