//
// Created by aapr6 on 1/12/25.
//

#include <gtest/gtest.h>
#include "../src/Metrics/F1Score.h"

TEST(F1ScoreTest, ComputeSimpleCase)
{
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

    EXPECT_DOUBLE_EQ(result, 1.0);
}

TEST(F1ScoreTest, ComputeWithFalsePositivesAndNegatives)
{
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

    // TP = 3, FP = 1, FN = 1
    double precision = 3.0 / (3.0 + 1.0);
    double recall = 3.0 / (3.0 + 1.0);
    double expected = 2.0 * (precision * recall) / (precision + recall);

    EXPECT_NEAR(result, expected, 1e-6);
}

TEST(F1ScoreTest, ComputeAllIncorrect)
{
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

    // TP = 0, FP = 0, FN = 3 => F1 = 0
    EXPECT_DOUBLE_EQ(result, 0.0);
}