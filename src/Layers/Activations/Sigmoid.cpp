//
// Created by aapr6 on 1/11/25.
//

#include "Sigmoid.h"

void Sigmoid::activate(uwu::Vector &output)
{
    output.iterate([&](double value)
    {
        return 1.0 / (1.0 + std::exp(-value));
    });
}

void Sigmoid::derivative(uwu::Vector &output)
{
    output.iterate([&](double value)
    {
        double sigmoid = 1.0 / (1.0 + std::exp(value));
        return sigmoid * (1.0 - sigmoid);
    });
}

Sigmoid *Sigmoid::clone() const {
    return new Sigmoid(*this);
}