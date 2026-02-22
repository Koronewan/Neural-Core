//
// Created by korone on 1/10/25.
//

#include "RidgeRegression.h"

#include <cmath>

double RidgeRegression::compute(Matrix &weights) const
{
    double sum = 0.0;
    weights.iterate([&sum](double& value, int i, int j) {
        sum += std::pow(value, 2.0);
    });

    return sum * this->alpha_;
}
