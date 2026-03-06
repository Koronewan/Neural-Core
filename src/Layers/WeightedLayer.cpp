//
// Created by korone on 12/21/24.
//

#include "WeightedLayer.h"

#include <iostream>
#include <ostream>

#include "MathUtils/Matrix.h"

WeightedLayer::WeightedLayer(const int numIn, const int numOut,
    InterfaceActivationFunction *activationFunction, InterfaceInitializer *weightsInitializer,
    Regularizer *regularizer)
{
    this->activationFunction_ = activationFunction;
    this->weights_ = Matrix(numOut, numIn);
    this->biases_ = uwu::Vector(numOut);
    this->regularizer_ = regularizer;

    weightsInitializer->initialize(weights_, biases_);

    this->z_ = uwu::Vector(numOut);
    this->biasesGradient_ = uwu::Vector(numOut);
    this->weightsGradient_ = Matrix(numOut, numIn);
}

WeightedLayer::WeightedLayer(InterfaceActivationFunction *activationFunction,
    const Matrix &weights, const uwu::Vector &biases)
{
    this->activationFunction_ = activationFunction;
    this->weights_ = weights;
    this->biases_ = biases;

    this->biasesGradient_ = uwu::Vector(weights.rows());
    this->weightsGradient_ = Matrix(weights.rows(), weights.columns());
}

uwu::Vector WeightedLayer::forward(const uwu::Vector &input)
{
    uwu::Vector output = uwu::Vector::dotProduct(this->weights_, input);
    output += this->biases_;

    this->z_ = output;
    this->activationFunction_->activate(output);
    return output;
}

void WeightedLayer::backward(uwu::Vector &error, const uwu::Vector &previousActivation)
{
    this->activationFunction_->derivative(this->z_);
    error *= this->z_;

    this->biasesGradient_ += error;
    this->weightsGradient_ += Matrix::outerProduct(error, previousActivation);
    error = uwu::Vector::dotProduct(this->weights_.transpose(), error);

    this->gradientCounter++;
}

void WeightedLayer::update(const std::string &event)
{
    if (event == "BatchStart")
    {
        this->biasesGradient_ = uwu::Vector(this->weights_.rows());
        this->weightsGradient_ = Matrix(this->weights_.rows(), this->weights_.columns());

        this->gradientCounter = 0;
    }
    else if (event == "BatchEnd")
    {
        const double regularizationValue = this->regularizer_->compute(this->weights_);
        this->weightsGradient_ += regularizationValue;
    }
}

std::string WeightedLayer::getInfo() const {
    std::ostringstream oss;
    oss << "Layer Type: WeightedLayer\n";
    oss << "Weights (Size: " << weights_.rows() << "x" << weights_.columns() << "):\n";
    oss << weights_.toString() << "\n";  // Imprime la matriz de pesos
    oss << "Weights Gradient (Size: " << weightsGradient_.rows() << "x" << weightsGradient_.columns() << "):\n";
    oss << weightsGradient_.toString() << "\n";  // Imprime la matriz del gradiente de pesos
    oss << "Biases (Size: " << biases_.size() << "):\n";
    oss << biases_.toString() << "\n";
    oss << "Biases Gradient (Size: " << biasesGradient_.size() << "):\n";
    oss << biasesGradient_.toString() << "\n";
    oss << "Activation Function: " << activationFunction_->getType() << "\n";
    oss << "Gradient Counter: " << gradientCounter << "\n";
    return oss.str();
}

