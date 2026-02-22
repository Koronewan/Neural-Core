//
// Created by aapr6 on 1/11/25.
//

#ifndef TANH_H
#define TANH_H

#include <vector>
#include <cmath>
#include "InterfaceActivationFunction.h"

class Tanh final : public InterfaceActivationFunction
{
public:
    void activate(uwu::Vector &output) override;
    void derivative(uwu::Vector &output) override;
    [[nodiscard]] Tanh* clone() const override;
    [[nodiscard]] std::string getType() const override { return "Tanh"; }
};

#endif //TANH_H
