//
// Created by korone on 1/10/25.
//

#ifndef UWU_LEARNER_REGULARIZER_H
#define UWU_LEARNER_REGULARIZER_H
#include <vector>
#include <string>

#include "MathUtils/Matrix.h"

class Regularizer
{
public:
    virtual ~Regularizer() = default;
    [[nodiscard]] virtual Regularizer* clone() const { return new Regularizer(*this); }
    virtual double compute(Matrix &weights) const;
    [[nodiscard]] std::string getType() { return "Regularizer"; }
};

#endif //UWU_LEARNER_REGULARIZER_H
