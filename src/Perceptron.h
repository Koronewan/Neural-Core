#ifndef UWU_LEARNER_PERCEPTRON_H
#define UWU_LEARNER_PERCEPTRON_H

#include "MathUtils/Vector.h"
#include "DataSet.h"
#include <iostream>
#include <random>

class Perceptron {
private:
    uwu::Vector weights_;
    double bias_;
    double learningRate_;

    double forwardPass(const uwu::Vector& item);
    void backpropagation(const uwu::Vector& item, double expected);

public:
    Perceptron();
    Perceptron(int inputSize, double learningRate);
    Perceptron(const Perceptron& other);
    void fit(const DataSet& dataSet, int epochs);
    uwu::Vector predict(const std::vector<std::vector<double>>& items);
    uwu::Vector getWeights() const;
};

#endif // UWU_LEARNER_PERCEPTRON_H