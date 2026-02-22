//
// Created by korone on 1/10/25.
//

#ifndef UWU_LEARNER_RIDGEREGRESSION_H
#define UWU_LEARNER_RIDGEREGRESSION_H
#include "Regularizer.h"
#include "MathUtils/Matrix.h"

class RidgeRegression: virtual public Regularizer
{
    double alpha_ = 0.01;
public:
    RidgeRegression() = default;
    explicit RidgeRegression(const double alpha): alpha_(alpha) {}
    double compute(Matrix &weights) const override;
    [[nodiscard]] RidgeRegression* clone() const override { return new RidgeRegression(*this); }
    [[nodiscard]] std::string getType() { return "Ridge"; }
};

#endif //UWU_LEARNER_RIDGEREGRESSION_H
