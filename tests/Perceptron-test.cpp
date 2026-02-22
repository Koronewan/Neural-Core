#include <gtest/gtest.h>
#include <random>
#include "../src/Perceptron.h"
#include "../src/DataSet.h"

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

    std::vector<std::vector<double>> labels_data_vector = {
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
    DataSet dataSet(features, labels_data_vector);

    double learningRate = 1.0;
    Perceptron perceptron(2, learningRate);

    perceptron.fit(dataSet, 1000);

    uwu::Vector predictions = perceptron.predict(features);

    int correctPredictions = 0;
    for (std::size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i] == labels[i]) {
            correctPredictions++;
        }
    }

    double accuracy = static_cast<double>(correctPredictions) / predictions.size();
    std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;

    EXPECT_GE(accuracy, 0.6);
}

TEST(PerceptronTest, LineSeparation) {
    // Generar datos de entrenamiento
    std::vector<std::vector<double>> features;
    std::vector<std::vector<double>> labels_data_vector;
    std::vector<double> labels;

    std::mt19937 generator(42);
    std::uniform_real_distribution<double> distribution(-10.0, 10.0);

    for (int i = 0; i < 100; ++i) {
        double x = distribution(generator);
        double y = distribution(generator);

        // y = 2x + 1
        double l = (y > 2 * x + 1) ? 1.0 : 0.0;
        std::vector<double> label_data = {l};
        std::vector<double> feature_data = {x, y};

        features.emplace_back(feature_data);
        labels_data_vector.push_back(label_data);
        labels.push_back(l);
    }

    DataSet dataSet(features, labels_data_vector);

    double learningRate = 0.1;
    Perceptron perceptron(2, learningRate);

    perceptron.fit(dataSet, 1000);

    uwu::Vector predictions = perceptron.predict(features);

    int correctPredictions = 0;
    for (std::size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i] == labels[i]) {
            correctPredictions++;
        }
    }

    double accuracy = static_cast<double>(correctPredictions) / features.size();
    EXPECT_GE(accuracy, 0.95);
}



