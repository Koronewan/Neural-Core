#include <gtest/gtest.h>
#include "../src/Layers/Dropout.h"

namespace {
    constexpr double HALF_DROPOUT_RATE = 0.5;
    constexpr double NO_DROPOUT_RATE = 0.0;
    constexpr double FULL_DROPOUT_RATE = 1.0;
    constexpr double DROPOUT_RATIO_TOLERANCE = 0.2;

    const std::vector<double> SAMPLE_INPUT = {1.0, 2.0, 3.0, 4.0, 5.0};
    const std::vector<double> SAMPLE_ERROR = {0.1, 0.2, 0.3, 0.4, 0.5};
}

TEST(DropoutTest, BackwardRespectsMask) {
    Dropout dropout(HALF_DROPOUT_RATE);

    uwu::Vector input(SAMPLE_INPUT);
    uwu::Vector forwardOutput = dropout.forward(input);

    uwu::Vector error(SAMPLE_ERROR);
    uwu::Vector originalError = error;

    dropout.backward(error, input);

    ASSERT_EQ(error.size(), forwardOutput.size());

    for (std::size_t i = 0; i < error.size(); ++i) {
        if (forwardOutput[i] == 0.0) {
            EXPECT_EQ(error[i], 0.0) << "Error should be zeroed where forward dropped neurons";
        } else {
            EXPECT_EQ(error[i], originalError[i]) << "Error should pass through where forward kept neurons";
        }
    }
}

TEST(DropoutTest, NoDropoutAtZeroRatio) {
    Dropout dropout(NO_DROPOUT_RATE);

    uwu::Vector input(SAMPLE_INPUT);
    uwu::Vector output = dropout.forward(input);

    ASSERT_EQ(input.size(), output.size());

    for (std::size_t i = 0; i < input.size(); ++i) {
        EXPECT_EQ(output[i], input[i]) << "No neurons should be dropped at zero rate";
    }
}

TEST(DropoutTest, FullDropoutAtOneRatio) {
    Dropout dropout(FULL_DROPOUT_RATE);

    uwu::Vector input(SAMPLE_INPUT);
    uwu::Vector output = dropout.forward(input);

    ASSERT_EQ(input.size(), output.size());

    for (std::size_t i = 0; i < input.size(); ++i) {
        EXPECT_EQ(output[i], 0.0) << "All outputs should be zero at full dropout";
    }
}
