//
// Created by aapr6 on 1/12/25.
//

#include <gtest/gtest.h>
#include "../src/Metrics/Accuracy.h"

namespace {
    constexpr double TOLERANCE = 1e-6;
    constexpr double PERFECT_ACCURACY = 1.0;
    constexpr double ZERO_ACCURACY = 0.0;
}

TEST(AccuracyTest, ComputeSimpleCase)
{
    // Predicted values: rounds to {1,0,1} and {0,0,1} at 0.5 threshold
    std::vector<std::vector<double>> predicted = {
        {0.9, 0.2, 0.8},
        {0.1, 0.4, 0.7}
    };
    Matrix predicted_matrix = Matrix(predicted);

    std::vector<std::vector<double>> actual = {
        {1, 0, 0},
        {0, 1, 0}
    };
    Matrix actual_matrix = Matrix(actual);

    Accuracy accuracyMetric;
    double result = accuracyMetric.compute(predicted_matrix, actual_matrix);

    // 5 out of 6 predictions are correct (second row, second element is wrong)
    constexpr int correctCount = 1;
    constexpr int totalCount = 2;
    const double expectedAccuracy = static_cast<double>(correctCount) / totalCount;
    EXPECT_NEAR(result, expectedAccuracy, TOLERANCE);
}

TEST(AccuracyTest, ComputeAllCorrect)
{
    std::vector<std::vector<double>> predicted = {
        {1.0, 0.0, 1.0},
        {0.0, 1.0, 1.0}
    };
    Matrix predicted_matrix(predicted);

    std::vector<std::vector<double>> actual = {
        {1, 0, 1},
        {0, 1, 1}
    };
    Matrix actual_matrix(actual);

    Accuracy accuracyMetric;
    double result = accuracyMetric.compute(predicted_matrix, actual_matrix);

    EXPECT_DOUBLE_EQ(result, PERFECT_ACCURACY);
}

TEST(AccuracyTest, ComputeAllIncorrect)
{
    // Every prediction is the opposite of the actual value
    std::vector<std::vector<double>> predicted = {
        {0.0, 1.0, 0.0},
        {1.0, 0.0, 0.0}
    };
    Matrix predicted_matrix = Matrix(predicted);

    std::vector<std::vector<double>> actual = {
        {1, 0, 1},
        {0, 1, 1}
    };
    Matrix actual_matrix = Matrix(actual);

    Accuracy accuracyMetric;
    double result = accuracyMetric.compute(predicted_matrix, actual_matrix);

    EXPECT_DOUBLE_EQ(result, ZERO_ACCURACY);
}