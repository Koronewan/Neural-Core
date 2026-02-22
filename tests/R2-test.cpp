//
// Created by korone on 1/11/25.
//

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include "../src/Metrics/R2.h"

/**
 * Helper function: compute the R² for a single column j, to compare
 * with your multi-output method’s column-by-column approach.
 */
double computeSingleColumnR2(const std::vector<std::vector<double>>& predicted,
                             const std::vector<std::vector<double>>& actual,
                             int colIndex)
{
    double ssr = 0.0; // sum of squared residuals
    double sst = 0.0; // total sum of squares

    // 1) Compute mean of the actual for this column
    double meanActual = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        meanActual += actual[i][colIndex];
    }
    meanActual /= static_cast<double>(actual.size());

    // 2) Compute SSR & SST
    for (size_t i = 0; i < predicted.size(); ++i) {
        double resid = actual[i][colIndex] - predicted[i][colIndex];
        ssr += resid * resid;

        double dev = actual[i][colIndex] - meanActual;
        sst += dev * dev;
    }

    // R² for that single column
    return 1.0 - (ssr / sst);
}

/**
 * Test 1: Perfect prediction across multiple columns.
 * Each column’s R² = 1. Summation => #columns, then we divide by #columns => 1.0.
 */
TEST(R2MultiOutputTest, PerfectPrediction) {
    R2 r2;

    // 3 samples (rows), 2 neurons (columns)
    // Perfect => predicted == actual
    std::vector<std::vector<double>> actual = {
        {1.0,  5.0},
        {2.0,  6.0},
        {3.0,  7.0}
    };
    Matrix actualMatrix(actual);

    std::vector<std::vector<double>> predicted = actual; // identical
    Matrix predictedMatrix(predicted);

    // If each column’s R² = 1, sum = 2, then / 2 columns => final = 1
    double score = r2.compute(predictedMatrix, actualMatrix);
    EXPECT_DOUBLE_EQ(score, 1.0)
        << "Perfect prediction for all columns should yield R² = 1.0.";
}

/**
 * Test 2: Slight offset for each column => R² less than 1.
 * We'll manually compute column-wise R², sum them, then /2 columns.
 */
TEST(R2MultiOutputTest, PartialOffsetMultipleColumns) {
    R2 r2;

    // 3 samples, 2 columns
    std::vector<std::vector<double>> actual = {
        {1.0,  5.0},
        {2.0,  6.0},
        {3.0,  7.0}
    };
    Matrix actualMatrix(actual);

    // Introduce small offsets for each column
    // e.g., predicted = actual + {+0.1, -0.2}
    std::vector<std::vector<double>> predicted = {
        {1.1,  4.8},
        {2.1,  5.8},
        {3.1,  6.8}
    };
    Matrix predictedMatrix(predicted);

    // Manually compute R² for col=0 and col=1
    double r2col0 = computeSingleColumnR2(predicted, actual, 0);
    double r2col1 = computeSingleColumnR2(predicted, actual, 1);

    // Your method sums the R² values and divides by #columns=2
    double expectedScore = (r2col0 + r2col1) / 2.0;

    double score = r2.compute(predictedMatrix, actualMatrix);
    EXPECT_NEAR(score, expectedScore, 1e-12)
        << "R² for multi-column data should match the manual column-by-column calculation.";
}

/**
 * Test 3: Very poor prediction => each column’s R² might be negative.
 * We verify that the final average can also be negative.
 */
TEST(R2MultiOutputTest, NegativeR2Example) {
    R2 r2;

    // 2 samples, 2 columns
    std::vector<std::vector<double>> actual = {
        {1.0,  10.0},
        {2.0,  20.0}
    };
    Matrix actualMatrix(actual);

    // Predict totally off
    std::vector<std::vector<double>> predicted = {
        {0.0,  0.0},
        {0.0,  0.0}
    };
    Matrix predictedMatrix(predicted);

    // Manually compute each column’s R²
    double r2col0 = computeSingleColumnR2(predicted, actual, 0);
    double r2col1 = computeSingleColumnR2(predicted, actual, 1);

    double expectedScore = (r2col0 + r2col1) / 2.0; // #columns = 2

    double score = r2.compute(predictedMatrix, actualMatrix);
    EXPECT_DOUBLE_EQ(score, expectedScore)
        << "R² can be negative if model is worse than a naive mean-based predictor.";
}