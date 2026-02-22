//
// Created by aapr6 on 1/10/25.
//

#include "Adam.h"

#include <iostream>
#include <ostream>

Adam::Adam(double learningRate, double beta1, double beta2, double epsilon)
{
    this->learningRate_ = learningRate;
    this->beta1_ = beta1;
    this->beta2_ = beta2;
    this->epsilon_ = epsilon;
}

void Adam::update(const uwu::Vector &gradient, uwu::Vector &bias, OptimizerState &state)
{
    auto& adamState = dynamic_cast<AdamState&>(state);

    // Inicializar los vectores mWeight y vWeight si están vacíos
    if (adamState.momentumBias.empty())
    {
        adamState.momentumBias.resize(bias.size(), 0.0);
        adamState.velocitiesBias.resize(bias.size(), 0.0);
    }

    ++adamState.t;

    const double alpha = learningRate_ * std::sqrt(1.0 - std::pow(beta2_, adamState.t))
        / (1.0 - std::pow(beta1_, adamState.t));

    adamState.momentumBias += (gradient - adamState.momentumBias) * (1.0 - beta1_);
    adamState.velocitiesBias += ((gradient ^ 2.0) - adamState.velocitiesBias) * (1.0 - beta2_);

    uwu::Vector correctedVelocities = adamState.velocitiesBias;
    correctedVelocities.iterate([&](const double val)->double {
        return std::sqrt(val) + this->epsilon_;
    });

    bias -= (adamState.momentumBias * alpha) / correctedVelocities;
}

void Adam::update(const Matrix& gradient, Matrix& weights, OptimizerState& state)
{
    auto& adamState = dynamic_cast<AdamState&>(state);

    // Verificar si los vectores mWeight y vWeight están vacíos, e inicializarlos según las dimensiones de la matriz
    if (adamState.momentumWeights.rows() == 0)
    {
        adamState.momentumWeights = Matrix(gradient.rows(), gradient.columns(), 0.0);
        adamState.velocitiesWeights = Matrix(gradient.rows(), gradient.columns(), 0.0);
    }

    const double alpha = learningRate_ * std::sqrt(1.0 - std::pow(beta2_, adamState.t))
        / (1.0 - std::pow(beta1_, adamState.t));

    adamState.momentumWeights += (gradient - adamState.momentumWeights) * (1.0 - beta1_);
    adamState.velocitiesWeights += ((gradient ^ 2.0) - adamState.velocitiesWeights) * (1.0 - beta2_);

    Matrix correctedVelocities = adamState.velocitiesWeights;
    correctedVelocities.iterate([&](const double val)->double {
        return std::sqrt(val) + this->epsilon_;
    });

    weights -= (adamState.momentumWeights * alpha) / correctedVelocities;
}