//
// Created by korone on 12/21/24.
//

#ifndef NEURAL_CORE_INTERFACEOPTIMIZER_H
#define NEURAL_CORE_INTERFACEOPTIMIZER_H
#include "OptimizerState.h"
#include <vector>
#include "MathUtils/Matrix.h"

class InterfaceOptimizer
{
public:
    virtual void update(const uwu::Vector& gradient,
        uwu::Vector &bias, OptimizerState &state) = 0;
    virtual void update(const Matrix& gradient,
        Matrix &weights, OptimizerState &state) = 0;
    virtual InterfaceOptimizer* clone() const = 0;

    virtual ~InterfaceOptimizer() = default;

};

#endif //NEURAL_CORE_INTERFACEOPTIMIZER_H
