//
// Created by korone on 12/21/24.
//

#ifndef DENSE_H
#define DENSE_H

#include "InterfaceLayer.h"
#include "Activations/InterfaceActivationFunction.h"
#include "Activations/ReLU.h"
#include "Activations/Sigmoid.h"
#include "Activations/Tanh.h"
#include "Regularization/LassoRegression.h"
#include "Regularization/LassoRidgeRegression.h"
#include "Regularization/RidgeRegression.h"
#include "Initializers/InterfaceInitializer.h"
#include "MathUtils/Matrix.h"
#include "Regularization/Regularizer.h"
#include "MathUtils/Vector.h"

class WeightedLayer final : public InterfaceLayer
{
    uwu::Vector z_;
    Matrix weights_;
    uwu::Vector biases_;
    Matrix weightsGradient_;
    uwu::Vector biasesGradient_;
    InterfaceActivationFunction* activationFunction_{};
    Regularizer* regularizer_= new Regularizer();
    int gradientCounter = 0;

public:
    WeightedLayer() = default;
    WeightedLayer(int numIn, int numOut, InterfaceActivationFunction* activationFunction,
        InterfaceInitializer* weightsInitializer, Regularizer *regularizer = new Regularizer());

    WeightedLayer(InterfaceActivationFunction* activationFunction,
        const Matrix &weights, const uwu::Vector &biases);

    uwu::Vector forward(const uwu::Vector &input) override;
    void backward(uwu::Vector &error, const uwu::Vector &previousActivation) override;

    Matrix &getWeights()
    {
        return weights_;
    }

    Matrix getWeightsGradients()
    {
        return weightsGradient_ /= this->gradientCounter;
    }

    uwu::Vector &getBiases()
    {
        return biases_;
    }

    uwu::Vector getBiasesGradients()
    {
        return biasesGradient_ / this->gradientCounter;
    }

    void update(const std::string &event) override;

    ~WeightedLayer() override
    {
        delete activationFunction_; // Libera la memoria asignada dinámicamente
        delete regularizer_;        // Libera la memoria asignada dinámicamente
    }

    [[nodiscard]] std::string getType() const override {return "WeightedLayer";}

    void saveToBinary(std::ofstream &outFile) const override;
    void loadFromBinary(std::ifstream &inFile) override;
    [[nodiscard]] std::string getInfo() const override;
};

#endif //DENSE_H
