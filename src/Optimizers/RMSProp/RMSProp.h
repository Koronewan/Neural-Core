//
// Created by aapr6 on 1/10/25.
//

#ifndef RMSPROP_H
#define RMSPROP_H

#include "../InterfaceOptimizer.h"
#include "RMSPropState.h"
#include <cmath>

class RMSProp: public InterfaceOptimizer
{
    double learning_rate_;
    double gamma_;
    double epsilon_;

public:
    explicit RMSProp(double learning_rate = 0.001, double gamma = 0.9, double epsilon = 1e-8);
    void update(const uwu::Vector& gradient,
        uwu::Vector &bias, OptimizerState &state) override;
    void update(const Matrix& gradient,
        Matrix &weights, OptimizerState &state) override;
    [[nodiscard]] RMSProp* clone() const override{ return new RMSProp(*this); }
};

#endif //RMSPROP_H
