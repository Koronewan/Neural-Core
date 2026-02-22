//
// Created by aapr6 on 1/12/25.
//

#include <gtest/gtest.h>
#include "../src/Metrics/Accuracy.h"

TEST(AccuracyTest, ComputeSimpleCase)
{
    std::vector<std::vector<double>> predicted = {
        {0.9, 0.2, 0.8}, // Redondea a: {1, 0, 1}
        {0.1, 0.4, 0.7}  // Redondea a: {0, 0, 1}
    };
    Matrix predicted_matrix = Matrix(predicted);

    std::vector<std::vector<double>> actual = {
        {1, 0, 1},
        {0, 1, 1}
    };
    Matrix actual_matrix = Matrix(actual);

    Accuracy accuracyMetric;

    double result = accuracyMetric.compute(predicted_matrix, actual_matrix);

    double expected = 5.0 / 6.0;
    EXPECT_NEAR(result, expected, 1e-6);
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

    EXPECT_DOUBLE_EQ(result, 1.0);
}

TEST(AccuracyTest, ComputeAllIncorrect)
{
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

    EXPECT_DOUBLE_EQ(result, 0.0);
}