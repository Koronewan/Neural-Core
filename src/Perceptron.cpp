#include "Perceptron.h"

Perceptron::Perceptron()
    : learningRate_(0.1), weights_(2, 0.0), bias_(0.0) { // Inicializa con 2 pesos por defecto
    // Inicializa los pesos aleatoriamente entre -0.5 y 0.5
    std::mt19937 generator(42); // Semilla fija para reproducibilidad
    std::uniform_real_distribution<double> distribution(-0.5, 0.5);
    for (size_t i = 0; i < weights_.size(); ++i) {
        weights_[i] = distribution(generator);
    }
}

Perceptron::Perceptron(int inputSize, double learningRate)
    : learningRate_(learningRate), weights_(inputSize, 0.0), bias_(0.0) {
    std::mt19937 generator(42);
    std::uniform_real_distribution<double> distribution(-0.5, 0.5);
    for (size_t i = 0; i < weights_.size(); ++i) {
        weights_[i] = distribution(generator);
    }
}

Perceptron::Perceptron(const Perceptron& other) = default;

double Perceptron::forwardPass(const uwu::Vector& item) {
    uwu::Vector temp = weights_;
    temp *= item;

    double sum = 0.0;
    for (std::size_t i = 0; i < temp.size(); ++i) {
        sum += temp[i];
    }

    sum += bias_;

    return sum >= 0.0 ? 1.0 : 0.0;
}


void Perceptron::backpropagation(const uwu::Vector& item, double expected){
    double predicted = forwardPass(item);
    double error = expected - predicted;
    weights_ += item * learningRate_ * error;
    bias_ += learningRate_ * error;
}

void Perceptron::fit(const DataSet& dataSet, int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (int i = 0; i < dataSet.getItems(); ++i) {
            uwu::Vector feature(dataSet.getFeatures()[i]);
            double label = dataSet.getLabels()[i][0];
            backpropagation(feature, label);
        }
    }
}

uwu::Vector Perceptron::predict(const std::vector<std::vector<double>>& items)
{
    uwu::Vector predictions(items.size());
    for (std::size_t i = 0; i < items.size(); ++i) {
        predictions[i] = forwardPass(items[i]);
    }
    return predictions;
}

uwu::Vector Perceptron::getWeights() const {
    return weights_;
}