//
// Created by korone on 1/8/25.
//

#include <gtest/gtest.h>
#include "../src/Layers/Activations/ReLU.h"

namespace {
    constexpr double TOLERANCE = 1e-5;

    // Mix of negative and positive values to exercise both branches of ReLU
    const std::vector<double> TEST_INPUT = {
        -0.9705, -0.9054, 0.5125, -0.8614, 0.2941, 0.2524, -0.3645
    };
}

TEST(ReLUTest, ActivateFuncion)
{
    uwu::Vector input(TEST_INPUT);

    // ReLU: max(0, x) — negative values become 0, positive values pass through
    std::vector<double> expected = {0, 0, 0.5125, 0, 0.2941, 0.2524, 0};
    uwu::Vector expectedOutput = uwu::Vector(expected);

    ReLU relu;
    relu.activate(input);

    for (size_t i = 0; i < input.size(); ++i) {
        EXPECT_NEAR(input[i], expectedOutput[i], TOLERANCE);
    }
}

TEST(ReLUTest, DerivativeFuncion)
{
    uwu::Vector input = uwu::Vector(TEST_INPUT);

    // ReLU derivative: 0 for x<0, 1 for x>0
    std::vector<double> expected = {0, 0, 1, 0, 1, 1, 0};
    uwu::Vector expectedOutput = uwu::Vector(expected);

    ReLU relu;
    relu.derivative(input);

    EXPECT_EQ(input, expectedOutput);
}
