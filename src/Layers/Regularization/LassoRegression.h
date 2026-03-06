//
// Created by korone on 1/10/25.
//

#ifndef NEURAL_CORE_LASSOREGRESSION_H
#define NEURAL_CORE_LASSOREGRESSION_H
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

#endif //NEURAL_CORE_LASSOREGRESSION_H
