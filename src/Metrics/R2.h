//
// Created by korone on 1/11/25.
//

#ifndef UWU_LEARNER_R2_H
#define UWU_LEARNER_R2_H
#include "InterfaceMetric.h"

#include "MathUtils/Matrix.h"

class R2 final: public InterfaceMetric
{
public:
    double compute(const Matrix& predicted,
        const Matrix& actual) override;
};

#endif //UWU_LEARNER_R2_H
