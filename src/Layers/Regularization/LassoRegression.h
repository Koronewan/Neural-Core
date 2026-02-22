//
// Created by korone on 1/10/25.
//

#ifndef UWU_LEARNER_LASSOREGRESSION_H
#define UWU_LEARNER_LASSOREGRESSION_H
#include <string>

#include "Regularizer.h"
#include "MathUtils/Matrix.h"

class LassoRegression: virtual public Regularizer
{
    double alpha_ = 0.01;
public:
    LassoRegression() = default;
    explicit LassoRegression(const double alpha) : alpha_(alpha) {}
    double compute(Matrix &weights) const override;
    [[nodiscard]] LassoRegression* clone() const override { return new LassoRegression(*this); }
    [[nodiscard]] std::string getType() { return "Lasso"; }
};

#endif //UWU_LEARNER_LASSOREGRESSION_H
