//
// Created by aapr6 on 1/11/25.
//

#include "Tanh.h"

void Tanh::activate(uwu::Vector &output)
{
    output.iterate([&](double value)
    {
        return std::tanh(value);
    });
}

void Tanh::derivative(uwu::Vector &output)
{
    output.iterate([&](double value)
    {
        const double tanhValue = std::tanh(value);
        return 1.0 - tanhValue * tanhValue;
    });
}

Tanh* Tanh::clone() const {
    return new Tanh(*this);
}