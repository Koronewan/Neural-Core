//
// Created by korone on 1/7/25.
//

#include "ReLU.h"

void ReLU::activate(uwu::Vector &output)
{
    output.iterate([&](double value)
    {
        return std::max(0.0, value);
    });
}

void ReLU::derivative(uwu::Vector &output)
{
    output.iterate([&](double value)
    {
        return (value > 0.0) ? 1.0 : 0.0;
    });
}

ReLU* ReLU::clone() const {
    return new ReLU(*this);
}