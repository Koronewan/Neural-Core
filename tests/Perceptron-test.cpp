#include <gtest/gtest.h>
#include <random>
#include "../src/Perceptron.h"
#include "../src/DataSet.h"

namespace {
    constexpr int NUM_FEATURES = 2;
    constexpr int TRAINING_EPOCHS = 1000;
    constexpr double MIN_ACCURACY_BASIC = 0.6;
    constexpr double MIN_ACCURACY_LINEAR_SEPARATION = 0.95;
    constexpr unsigned int RANDOM_SEED = 42;

    // Data generation parameters for linear separation test
    constexpr double DATA_RANGE_MIN = -10.0;
    constexpr double DATA_RANGE_MAX = 10.0;
    constexpr int LINEAR_SEPARATION_SAMPLES = 100;

    // Decision boundary: y = BOUNDARY_SLOPE * x + BOUNDARY_INTERCEPT
    constexpr double BOUNDARY_SLOPE = 2.0;
    constexpr double BOUNDARY_INTERCEPT = 1.0;
}

TEST(PerceptronTest, TrainMoreThanFive) {
    std::vector<std::vector<double>> features = {
        {2.0, 4.0},
        {1.0, 1.0},
        {1.0, 2.0},
        {3.0, 2.0},
        {4.0, 3.0},
        {5.0, 3.0},
        {6.0, 4.0},
        {5.0, 5.0},
        {3.0, 2.0},
        {1.0, 4.0}
    };

    std::vector<std::vector<double>> labelsMatrix = {
        {1.0},
        {0.0},
        {0.0},
        {0.0},
        {1.0},
        {1.0},
        {1.0},
        {1.0},
        {0.0},
        {0.0}
    };
    std::vector<double> labels = {1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0};
    DataSet dataSet(features, labelsMatrix);

    constexpr double learningRate = 1.0;
    Perceptron perceptron(NUM_FEATURES, learningRate);

    perceptron.fit(dataSet, TRAINING_EPOCHS);

    uwu::Vector predictions = perceptron.predict(features);

    int correctPredictions = 0;
    for (std::size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i] == labels[i]) {
            correctPredictions++;
        }
    }

    double accuracy = static_cast<double>(correctPredictions) / predictions.size();
    std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;

    EXPECT_GE(accuracy, MIN_ACCURACY_BASIC);
}

TEST(PerceptronTest, LineSeparation) {
    std::vector<std::vector<double>> features;
    std::vector<std::vector<double>> labelsMatrix;
    std::vector<double> labels;

    std::mt19937 generator(RANDOM_SEED);
    std::uniform_real_distribution<double> distribution(DATA_RANGE_MIN, DATA_RANGE_MAX);

    for (int i = 0; i < LINEAR_SEPARATION_SAMPLES; ++i) {
        double x = distribution(generator);
        double y = distribution(generator);

        // Classify based on decision boundary: y > slope*x + intercept
        double label = (y > BOUNDARY_SLOPE * x + BOUNDARY_INTERCEPT) ? 1.0 : 0.0;
        features.emplace_back(std::vector<double>{x, y});
        labelsMatrix.push_back({label});
        labels.push_back(label);
    }

    DataSet dataSet(features, labelsMatrix);

    constexpr double learningRate = 0.1;
    Perceptron perceptron(NUM_FEATURES, learningRate);

    perceptron.fit(dataSet, TRAINING_EPOCHS);

    uwu::Vector predictions = perceptron.predict(features);

    int correctPredictions = 0;
    for (std::size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i] == labels[i]) {
            correctPredictions++;
        }
    }

    double accuracy = static_cast<double>(correctPredictions) / features.size();
    EXPECT_GE(accuracy, MIN_ACCURACY_LINEAR_SEPARATION);
}



