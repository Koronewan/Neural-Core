//
// Created by korone on 1/9/25.
//

#include "Layers/Initializers/HeInitializer.h"
#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include <cmath>

namespace {
    constexpr double STDDEV_RELATIVE_TOLERANCE = 0.2;  // 20% tolerance for statistical checks
    constexpr int MEAN_STANDARD_ERRORS = 5;            // Number of standard errors for mean check

    // Small matrix dimensions for basic tests
    constexpr int SMALL_NUM_OUTPUTS = 4;
    constexpr int SMALL_NUM_INPUTS = 3;

    // Medium matrix dimensions for dimension-checking test
    constexpr int MEDIUM_NUM_OUTPUTS = 5;
    constexpr int MEDIUM_NUM_INPUTS = 2;

    // Large matrix dimensions for statistical tests
    constexpr int LARGE_NUM_OUTPUTS = 50;
    constexpr int LARGE_NUM_INPUTS = 20;
}

/**
 * Helper function to compute the mean of a vector of numbers.
 */
double mean(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    return sum / values.size();
}

/**
 * Helper function to compute the (sample) standard deviation of a vector of numbers.
 */
double standardDeviation(const std::vector<double>& values) {
    if (values.size() < 2) return 0.0;
    double avg = mean(values);
    double variance = 0.0;
    for (double v : values) {
        double diff = v - avg;
        variance += diff * diff;
    }
    variance /= (values.size() - 1); // sample variance
    return std::sqrt(variance);
}

TEST(HeInitializerTest, BiasesSetToZero) {
    std::vector<std::vector<double>> weights(SMALL_NUM_OUTPUTS, std::vector<double>(SMALL_NUM_INPUTS));
    Matrix weightsMatrix(weights);

    std::vector<double> biases(SMALL_NUM_OUTPUTS);
    uwu::Vector biasesVector(biases);

    HeInitializer initializer;
    initializer.initialize(weightsMatrix, biasesVector);

    for (double b : biases) {
        EXPECT_DOUBLE_EQ(b, 0.0) << "Bias should be initialized to zero.";
    }
}

TEST(HeInitializerTest, WeightsHaveCorrectDimensions) {
    std::vector<std::vector<double>> weights(MEDIUM_NUM_OUTPUTS, std::vector<double>(MEDIUM_NUM_INPUTS));
    Matrix weightsMatrix(weights);

    std::vector<double> biases(MEDIUM_NUM_OUTPUTS);
    uwu::Vector biasesVector(biases);

    HeInitializer initializer;
    initializer.initialize(weightsMatrix, biasesVector);

    ASSERT_EQ(weightsMatrix.size(), MEDIUM_NUM_OUTPUTS);
    weightsMatrix.iterate([&](double& /*element*/, int row, int /*col*/) {
        ASSERT_EQ(weightsMatrix[row].size(), static_cast<size_t>(MEDIUM_NUM_INPUTS));
    });
    ASSERT_EQ(biasesVector.size(), static_cast<size_t>(MEDIUM_NUM_OUTPUTS));
}

TEST(HeInitializerTest, WeightsStatisticalCheck) {
    std::vector<std::vector<double>> weights(LARGE_NUM_OUTPUTS, std::vector<double>(LARGE_NUM_INPUTS));
    Matrix weightsMatrix(weights);

    std::vector<double> biases(LARGE_NUM_OUTPUTS);
    uwu::Vector biasesVector(biases);

    HeInitializer initializer;
    initializer.initialize(weightsMatrix, biasesVector);

    // Collect all weights for statistical analysis
    std::vector<double> allWeights;
    allWeights.reserve(LARGE_NUM_OUTPUTS * LARGE_NUM_INPUTS);
    weightsMatrix.iterate([&](double value, int i, int j) {
        allWeights.push_back(value);
    });

    // He initialization: expected stddev = sqrt(2 / nInputs)
    const double expectedStd = std::sqrt(2.0 / static_cast<double>(LARGE_NUM_INPUTS));

    double avg = mean(allWeights);
    double stddev = standardDeviation(allWeights);

    // Mean should be near zero (within a few standard errors)
    const double standardErrorOfMean = stddev / std::sqrt(allWeights.size());
    EXPECT_NEAR(avg, 0.0, MEAN_STANDARD_ERRORS * standardErrorOfMean)
        << "Mean of He-initialized weights should be near 0.";

    // Standard deviation should be near the theoretical value
    const double allowedDeviation = STDDEV_RELATIVE_TOLERANCE * expectedStd;
    EXPECT_NEAR(stddev, expectedStd, allowedDeviation)
        << "Std dev of He-initialized weights is outside the expected range.";
}
