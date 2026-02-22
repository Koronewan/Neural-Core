//
// Created by korone on 1/7/25.
//

#include "SGD.h"

void SGD::update(const uwu::Vector &gradient, uwu::Vector &bias, OptimizerState &state)
{
    bias -= gradient * this->learningRate_;
}

void SGD::update(const Matrix &gradient, Matrix &weights,
    OptimizerState &state)
{
    weights -= gradient * this->learningRate_;
}
