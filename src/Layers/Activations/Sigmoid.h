//
// Created by aapr6 on 1/11/25.
//

#ifndef SIGMOID_H
#define SIGMOID_H

#include "InterfaceActivationFunction.h"
#include <cmath>

class Sigmoid final: public InterfaceActivationFunction
{
public:
    void activate(uwu::Vector &output) override;
    void derivative(uwu::Vector &output) override;
    [[nodiscard]] Sigmoid* clone() const override;
    [[nodiscard]] std::string getType() const override { return "Sigmoid"; }
};

#endif //SIGMOID_H
