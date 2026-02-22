//
// Created by korone on 12/21/24.
//

#ifndef UWU_LEARNER_INTERFACEMETRIC_H
#define UWU_LEARNER_INTERFACEMETRIC_H

#include "MathUtils/Matrix.h"

class InterfaceMetric
{
public:
    virtual ~InterfaceMetric() = default;

    virtual double compute(const Matrix& predicted,
        const Matrix& actual) = 0;
};

#endif //UWU_LEARNER_INTERFACEMETRIC_H
