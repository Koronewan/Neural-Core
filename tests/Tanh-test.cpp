//
// Created by aapr6 on 1/11/25.
//

#include <gtest/gtest.h>
#include <cmath>
#include "../src/Layers/Activations/Tanh.h"

TEST(TanhTest, ActivateFunction)
{
    std::vector<double> vectorInput = {-1.0, -0.5, 0.0, 0.5, 1.0};
    uwu::Vector input = uwu::Vector(vectorInput);
    std::vector<double> expected = {
        -0.76159416,
        -0.4621171573,
        0.0,
        0.4621171573,
        0.761594156
    };
    uwu::Vector expectedOutput = uwu::Vector(vectorInput);

    Tanh tanh;
    tanh.activate(input);

    EXPECT_EQ(input, expectedOutput);
}

TEST(TanhTest, DerivativeFunction)
{
    std::vector<double> vectorInput = {-1.0, -0.5, 0.0, 0.5, 1.0};
    uwu::Vector input = uwu::Vector(vectorInput);
    std::vector<double> expected;

    for (double x : vectorInput)
    {
        double tanhValue = std::tanh(x);
        expected.push_back(1.0 - tanhValue * tanhValue);
    }
    uwu::Vector expectedOutput = uwu::Vector(expected);

    Tanh tanh;
    tanh.derivative(input);

    EXPECT_EQ(input, expectedOutput);
}