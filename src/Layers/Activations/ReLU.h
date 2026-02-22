//
// Created by korone on 1/7/25.
//

#ifndef UWU_LEARNER_RELU_H
#define UWU_LEARNER_RELU_H
#include "InterfaceActivationFunction.h"


class ReLU final: public InterfaceActivationFunction
{
public:
    void activate(uwu::Vector &output) override;
    void derivative(uwu::Vector &output) override;
    [[nodiscard]] ReLU* clone() const override;
    [[nodiscard]] std::string getType() const override { return "ReLU"; }
};



#endif //UWU_LEARNER_RELU_H
