//
// Created by korone on 1/9/25.
//

#include "Layers/Initializers/HeInitializer.h"
#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include <cmath>

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
    // Suppose we want a matrix of (nOutputs x nInputs), e.g. 4x3
    int nOutputs = 4;
    int nInputs = 3;

    std::vector<std::vector<double>> weights(nOutputs, std::vector<double>(nInputs));
    Matrix weightsMatrix(weights);

    std::vector<double> biases(nOutputs);
    uwu::Vector biasesVector(biases);

    HeInitializer initializer;
    initializer.initialize(weightsMatrix, biasesVector);

    // Check that all biases are exactly 0
    for (double b : biases) {
        EXPECT_DOUBLE_EQ(b, 0.0) << "Bias should be initialized to zero.";
    }
}

TEST(HeInitializerTest, WeightsHaveCorrectDimensions) {
    int nOutputs = 5;
    int nInputs = 2;

    std::vector<std::vector<double>> weights(nOutputs, std::vector<double>(nInputs));
    Matrix weightsMatrix(weights);

    std::vector<double> biases(nOutputs);
    uwu::Vector biasesVector(biases);

    HeInitializer initializer;
    initializer.initialize(weightsMatrix, biasesVector);

    // Check dimensions
    ASSERT_EQ(weightsMatrix.size(), nOutputs);
    weightsMatrix.iterate([&](double& /*element*/, int row, int /*col*/) {
        ASSERT_EQ(weightsMatrix[row].size(), static_cast<size_t>(nInputs));
    });
    ASSERT_EQ(biasesVector.size(), static_cast<size_t>(nOutputs));
}

TEST(HeInitializerTest, WeightsStatisticalCheck) {
    // We'll do a "larger" matrix to get a better sample of random values
    // for a basic statistical check (though for very large dimension,
    // it might cause longer test time).
    int nOutputs = 50;
    int nInputs = 20;

    std::vector<std::vector<double>> weights(nOutputs, std::vector<double>(nInputs));
    Matrix weightsMatrix(weights);

    std::vector<double> biases(nOutputs);
    uwu::Vector biasesVector(biases);

    HeInitializer initializer;
    initializer.initialize(weightsMatrix, biasesVector);


    // Collect all weights into a single vector for convenience
    std::vector<double> allWeights;
    allWeights.reserve(nOutputs * nInputs);
    weightsMatrix.iterate([&](double value, int i, int j) {
        allWeights.push_back(value);
    });

    // Expected standard deviation is sqrt(2 / nInputs)
    double expectedStd = std::sqrt(2.0 / static_cast<double>(nInputs));

    double avg = mean(allWeights);
    double stddev = standardDeviation(allWeights);

    // Because these are random, we allow some leeway for normal variation.
    // The larger the sample (nOutputs*nInputs), the closer these should be in theory.
    // A typical rule of thumb might be ~3*SE range for typical test "tolerance."
    // Standard Error of the mean for normal is (stddev / sqrt(N)).

    // 1) Mean should be near zero
    //    Let's require it be within ~5 std errors from 0 to be safe
    double SE_mean = stddev / std::sqrt(allWeights.size());
    EXPECT_NEAR(avg, 0.0, 5 * SE_mean)
        << "Mean of He-initialized weights should be near 0.";

    // 2) Standard deviation should be near expectedStd
    //    We'll allow 20% tolerance or so, as a simple check
    //    Tolerance can be tuned depending on sample size & desired strictness.
    double allowedDeviation = 0.2 * expectedStd;
    EXPECT_NEAR(stddev, expectedStd, allowedDeviation)
        << "Std dev of He-initialized weights is outside the expected range.";
}
