//
// Created by aapr6 on 1/12/25.
//

#include "gtest/gtest.h"
#include "../src/Loss/CrossEntropy.h"

namespace {
    constexpr double TOLERANCE = 1e-5;
    constexpr double NEAR_ZERO_PREDICTION = 1e-10;
    constexpr double CLAMP_GRADIENT_MAGNITUDE = 1e8;
}

TEST(CrossEntropyTest, GradientComputation)
{
    // Predictions and ground truth labels
    std::vector<double> predictions = {0.1, 0.9, 0.8};
    uwu::Vector item = uwu::Vector(predictions);

    std::vector<double> labels = {0.0, 1.0, 0.0};
    uwu::Vector expectedItem = uwu::Vector(labels);

    // Expected gradient: -label/prediction for each element
    // For label=0: gradient=0, For label=1, pred=0.9: gradient=-1/0.9 ~ -1.1111111
    std::vector<double> expectedGradient = {-0.0, -1.1111111, -0.0};

    CrossEntropy lossFunction;
    uwu::Vector computedGradient = lossFunction.gradient(item, expectedItem);

    for (size_t i = 0; i < computedGradient.size(); ++i) {
        EXPECT_NEAR(computedGradient[i], expectedGradient[i], TOLERANCE);
    }
}

// Predictions far from ground truth labels
TEST(CrossEntropyTest, GradientDifferentPredictions)
{
    std::vector<double> predictions = {0.05, 0.7, 0.2};
    uwu::Vector item = uwu::Vector(predictions);

    std::vector<double> labels = {1.0, 0.0, 1.0};
    uwu::Vector expectedItem = uwu::Vector(labels);

    // gradient = -label/prediction
    // pred=0.05, label=1: -1/0.05=-20; pred=0.7, label=0: 0; pred=0.2, label=1: -1/0.2=-5
    std::vector<double> expectedGradient = {-19.999996, 0.0, -4.99999999};

    CrossEntropy lossFunction;
    uwu::Vector computedGradient = lossFunction.gradient(item, expectedItem);

    for (int i = 0; i < computedGradient.size(); ++i) {
        EXPECT_NEAR(computedGradient[i], expectedGradient[i], TOLERANCE);
    }
}

// Near-zero predictions test numerical stability
TEST(CrossEntropyTest, GradientWithSmallPredictions)
{
    std::vector<double> predictions = {NEAR_ZERO_PREDICTION, NEAR_ZERO_PREDICTION, NEAR_ZERO_PREDICTION};
    uwu::Vector item = uwu::Vector(predictions);

    std::vector<double> labels = {1.0, 0.0, 1.0};
    uwu::Vector expectedItem = uwu::Vector(labels);

    // With clamping, gradient = -label / clamped_prediction
    const double expectedClampedGradient = -99009900.990099013;
    std::vector<double> expectedGradient = {expectedClampedGradient, 0, expectedClampedGradient};

    CrossEntropy lossFunction;
    uwu::Vector computedGradient = lossFunction.gradient(item, expectedItem);

    for (int i = 0; i < computedGradient.size(); ++i) {
        EXPECT_NEAR(computedGradient[i], expectedGradient[i], TOLERANCE);
    }
}

// Predictions near 1.0 (high confidence)
TEST(CrossEntropyTest, GradientWithPredictionsCloseToOne)
{
    std::vector<double> predictions = {0.999, 0.95, 0.999};
    uwu::Vector item = uwu::Vector(predictions);

    std::vector<double> labels = {1.0, 1.0, 0.0};
    uwu::Vector expectedItem = uwu::Vector(labels);

    // gradient = -label/prediction
    // pred=0.999, label=1: -1/0.999~-1.001; pred=0.95, label=1: -1/0.95~-1.0526; pred=0.999, label=0: 0
    std::vector<double> expectedGradient = {-1.001000991, -1.052631568, 0};

    CrossEntropy lossFunction;
    uwu::Vector computedGradient = lossFunction.gradient(item, expectedItem);

    for (int i = 0; i < computedGradient.size(); ++i) {
        EXPECT_NEAR(computedGradient[i], expectedGradient[i], TOLERANCE);
    }
}

// Zero predictions test clamping behavior
TEST(CrossEntropyTest, GradientWithPredictionsZero)
{
    std::vector<double> predictions = {0.0, 0.0, 0.0};
    uwu::Vector item = uwu::Vector(predictions);

    std::vector<double> labels = {1.0, 0.0, 1.0};
    uwu::Vector expectedItem = uwu::Vector(labels);

    // Gradient is clamped for zero predictions: -label / clamped_min
    std::vector<double> expectedGradient = {-CLAMP_GRADIENT_MAGNITUDE, 0.0, -CLAMP_GRADIENT_MAGNITUDE};

    CrossEntropy lossFunction;
    uwu::Vector computedGradient = lossFunction.gradient(item, expectedItem);

    for (int i = 0; i < computedGradient.size(); ++i) {
        EXPECT_NEAR(computedGradient[i], expectedGradient[i], TOLERANCE);
    }
}