//
// Created by korone on 12/21/24.
//

#ifndef NEURAL_CORE_INTERFACEMETRIC_H
#define NEURAL_CORE_INTERFACEMETRIC_H

#include "MathUtils/Matrix.h"

class InterfaceMetric
{
public:
    virtual ~InterfaceMetric() = default;

    virtual double compute(const Matrix& predicted,
        const Matrix& actual) = 0;
};

#endif //NEURAL_CORE_INTERFACEMETRIC_H
