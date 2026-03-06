//
// Created by aapr6 on 1/11/25.
//

#include <gtest/gtest.h>
#include <cmath>
#include "../src/Layers/Activations/Tanh.h"

namespace {
    // Standard test points spanning the negative-to-positive range
    const std::vector<double> TEST_INPUT = {-1.0, -0.5, 0.0, 0.5, 1.0};
}

TEST(TanhTest, ActivateFunction)
{
    uwu::Vector input = uwu::Vector(TEST_INPUT);
    std::vector<double> expected;
    for (double x : TEST_INPUT)
    {
        expected.push_back(std::tanh(x));
    }
    uwu::Vector expectedOutput = uwu::Vector(expected);

    Tanh tanh;
    tanh.activate(input);

    EXPECT_EQ(input, expectedOutput);
}

TEST(TanhTest, DerivativeFunction)
{
    uwu::Vector input = uwu::Vector(TEST_INPUT);

    // Tanh derivative: 1 − tanh²(x)
    std::vector<double> expected;
    for (double x : TEST_INPUT)
    {
        double tanhValue = std::tanh(x);
        expected.push_back(1.0 - tanhValue * tanhValue);
    }
    uwu::Vector expectedOutput = uwu::Vector(expected);

    Tanh tanh;
    tanh.derivative(input);

    EXPECT_EQ(input, expectedOutput);
}