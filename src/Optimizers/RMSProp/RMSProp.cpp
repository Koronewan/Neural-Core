//
// Created by aapr6 on 1/10/25.
//

#include "RMSProp.h"

RMSProp::RMSProp(double learning_rate, double gamma, double epsilon)
{
    this->learning_rate_ = learning_rate;
    this->gamma_ = gamma;
    this->epsilon_ = epsilon;
}

void RMSProp::update(const uwu::Vector& gradient, uwu::Vector& bias, OptimizerState& state)
{
    auto& rmsState = dynamic_cast<RMSPropState&>(state);

    // Inicializa el vector de mean_square si está vacío
    if (rmsState.meanSquareBias.empty())
    {
        rmsState.meanSquareBias.resize(bias.size(), 0.0);
    }

    rmsState.meanSquareBias = rmsState.meanSquareBias * gamma_ + (gradient ^ 2) * (1.0 - gamma_);

    uwu::Vector denominator = rmsState.meanSquareBias;
    denominator.iterate([&](const double val) ->double {
        return std::sqrt(val + epsilon_);
    });

    bias -= gradient * learning_rate_ / denominator;
}

void RMSProp::update(const Matrix& gradient, Matrix& weights, OptimizerState& state)
{
    auto& rmsState = dynamic_cast<RMSPropState&>(state);

    // Inicializa la matriz meanSquareWeights si está vacía
    if (rmsState.meanSquareWeights.rows() == 0)
    {
        rmsState.meanSquareWeights = Matrix(gradient.rows(), gradient.columns(), 0.0);
    }

    rmsState.meanSquareWeights = rmsState.meanSquareWeights * gamma_ + (gradient ^ 2) * (1.0 - gamma_);

    Matrix denominator = rmsState.meanSquareWeights;
    denominator.iterate([&](const double val) ->double {
        return std::sqrt(val + epsilon_);
    });

    weights -= gradient * learning_rate_ / denominator;
}
