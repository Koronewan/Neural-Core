//
// Created by aapr6 on 1/12/25.
//

#include <gtest/gtest.h>
#include "../src/Metrics/F1Score.h"

namespace {
    constexpr double TOLERANCE = 1e-6;
    constexpr double PERFECT_SCORE = 1.0;
    constexpr double ZERO_SCORE = 0.0;
}

TEST(F1ScoreTest, ComputeSimpleCase)
{
    // Perfect predictions: all match actual labels
    std::vector<std::vector<double>> predicted = {
        {1.0, 0.0, 1.0},
        {0.0, 1.0, 1.0}
    };
    Matrix predictedMatrix = Matrix(predicted);

    std::vector<std::vector<double>> actual = {
        {1.0, 0, 1},
        {0, 1, 1}
    };
    Matrix actualMatrix = Matrix(actual);

    F1Score f1Metric;
    double result = f1Metric.compute(predictedMatrix, actualMatrix);

    EXPECT_DOUBLE_EQ(result, PERFECT_SCORE);
}

TEST(F1ScoreTest, ComputeWithFalsePositivesAndNegatives)
{
    // Predictions with one false positive and one false negative
    std::vector<std::vector<double>> predicted = {
        {1.0, 1.0, 0.0},
        {0.0, 1.0, 1.0}
    };
    Matrix predictedMatrix = Matrix(predicted);

    std::vector<std::vector<double>> actual = {
        {1, 0, 1},
        {0, 1, 1}
    };
    Matrix actualMatrix = Matrix(actual);

    F1Score f1Metric;
    double result = f1Metric.compute(predictedMatrix, actualMatrix);

    // TP=3, FP=1, FN=1 -> precision=3/4, recall=3/4
    constexpr int truePositives = 3;
    constexpr int falsePositives = 1;
    constexpr int falseNegatives = 1;
    const double precision = static_cast<double>(truePositives) / (truePositives + falsePositives);
    const double recall = static_cast<double>(truePositives) / (truePositives + falseNegatives);
    const double expectedF1 = 2.0 * (precision * recall) / (precision + recall);

    EXPECT_NEAR(result, expectedF1, TOLERANCE);
}

TEST(F1ScoreTest, ComputeAllIncorrect)
{
    // All predictions are zero, but actual has positives
    std::vector<std::vector<double>> predicted = {
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0}
    };
    Matrix predictedMatrix = Matrix(predicted);

    std::vector<std::vector<double>> actual = {
        {1, 0, 1},
        {0, 1, 1}
    };
    Matrix actualMatrix = Matrix(actual);

    F1Score f1Metric;
    double result = f1Metric.compute(predictedMatrix, actualMatrix);

    // TP=0, FP=0, FN=3 -> F1=0
    EXPECT_DOUBLE_EQ(result, ZERO_SCORE);
}