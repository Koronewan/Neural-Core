//
// Created by aapr6 on 1/10/25.
//

#ifndef ADAM_H
#define ADAM_H

#include "../InterfaceOptimizer.h"
#include "AdamState.h"
#include <cmath>
#include "MathUtils/Vector.h"

class Adam: public InterfaceOptimizer
{
    double learningRate_;
    double beta1_;
    double beta2_;
    double epsilon_;
public:
    explicit Adam(double learningRate = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-7);
    void update(const uwu::Vector& gradient,
        uwu::Vector &bias, OptimizerState &state) override;
    void update(const Matrix& gradient,
        Matrix &weights, OptimizerState &state) override;
    [[nodiscard]] Adam* clone() const override { return new Adam(*this); }
};

#endif //ADAM_H
