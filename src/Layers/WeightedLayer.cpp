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

void WeightedLayer::saveToBinary(std::ofstream &outFile) const {
    // Guardar el tipo de capa
    std::string layerType = "WeightedLayer";
    size_t typeLength = layerType.size();
    outFile.write(reinterpret_cast<const char*>(&typeLength), sizeof(typeLength));
    outFile.write(layerType.data(), typeLength);

    // Guardar pesos y biases
    weights_.saveToBinary(outFile); // Supongamos que Matrix tiene saveToBinary
    biases_.saveToBinary(outFile);  // Supongamos que Vector tiene saveToBinary
    weightsGradient_.saveToBinary(outFile);
    biasesGradient_.saveToBinary(outFile);

    // Guardar el tipo de función de activación
    std::string activationType = activationFunction_->getType(); // Ejemplo: "ReLU", "Sigmoid", "Tanh"
    size_t activationLength = activationType.size();
    outFile.write(reinterpret_cast<const char*>(&activationLength), sizeof(activationLength));
    outFile.write(activationType.data(), activationLength);

    // Guardar el regularizador
    std::string regularizerType = regularizer_->getType(); // Ejemplo: "Lasso", "Ridge", "LassoRidge"
    size_t regularizerLength = regularizerType.size();
    outFile.write(reinterpret_cast<const char*>(&regularizerLength), sizeof(regularizerLength));
    outFile.write(regularizerType.data(), regularizerLength);

    outFile.write(reinterpret_cast<const char*>(&gradientCounter), sizeof(gradientCounter));
}


void WeightedLayer::loadFromBinary(std::ifstream& inFile) {
    // Leer pesos y biases
    weights_.loadFromBinary(inFile); // Supongamos que Matrix tiene loadFromBinary
    biases_.loadFromBinary(inFile);  // Supongamos que Vector tiene loadFromBinary
    weightsGradient_.loadFromBinary(inFile);
    biasesGradient_.loadFromBinary(inFile);

    // Leer el tipo de función de activación
    size_t activationLength;
    inFile.read(reinterpret_cast<char*>(&activationLength), sizeof(activationLength));
    if (!inFile) throw std::runtime_error("Error al leer el tamaño del tipo de función de activación");

    std::string activationType(activationLength, '\0');
    inFile.read(&activationType[0], activationLength);
    if (!inFile) throw std::runtime_error("Error al leer el tipo de función de activación");

    // Crear la función de activación correspondiente
    delete activationFunction_; // Liberar memoria previa (si la hay)
    if (activationType == "ReLU") {
        activationFunction_ = new ReLU();
    } else if (activationType == "Sigmoid") {
        activationFunction_ = new Sigmoid();
    } else if (activationType == "Tanh") {
        activationFunction_ = new Tanh();
    } else {
        throw std::runtime_error("Tipo de función de activación desconocido: " + activationType);
    }

    // Leer el tipo de regularizador
    size_t regularizerLength;
    inFile.read(reinterpret_cast<char*>(&regularizerLength), sizeof(regularizerLength));
    if (!inFile) throw std::runtime_error("Error al leer el tamaño del tipo de regularizador");

    std::string regularizerType(regularizerLength, '\0');
    inFile.read(&regularizerType[0], regularizerLength);
    if (!inFile) throw std::runtime_error("Error al leer el tipo de regularizador");

    // Crear el regularizador correspondiente
    delete regularizer_; // Liberar memoria previa (si la hay)
    if (regularizerType == "Lasso") {
        regularizer_ = new LassoRegression();
    } else if (regularizerType == "Ridge") {
        regularizer_ = new RidgeRegression();
    } else if (regularizerType == "LassoRidge") {
        regularizer_ = new LassoRidgeRegression();
    } else if (regularizerType == "Regularizer"){
        regularizer_ = new Regularizer();
    }
    else {
        throw std::runtime_error("Tipo de regularizador desconocido: " + regularizerType);
    }

    // Leer el contador de gradientes
    inFile.read(reinterpret_cast<char*>(&gradientCounter), sizeof(gradientCounter));
    if (!inFile) throw std::runtime_error("Error al leer el contador de gradientes");
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

