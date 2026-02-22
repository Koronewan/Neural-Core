//
// Created by korone on 1/7/25.
//

#ifndef SGD_H
#define SGD_H

#include "../InterfaceOptimizer.h"
#include "MathUtils/Vector.h"
#include "SGDState.h"


class SGD: public InterfaceOptimizer
{
    double learningRate_;
public:
    explicit SGD(const double learningRate = 0.01): learningRate_(learningRate) {};
    void update(const uwu::Vector& gradient,
        uwu::Vector &bias, OptimizerState &state) override;
    void update(const Matrix& gradient,
        Matrix &weights, OptimizerState &state) override;
    [[nodiscard]] SGD* clone() const override { return new SGD(*this); }
};

#endif //SGD_H
