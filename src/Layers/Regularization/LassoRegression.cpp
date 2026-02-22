//
// Created by korone on 1/10/25.
//

#include "LassoRegression.h"

#include <cmath>

double LassoRegression::compute(Matrix &weights) const
{
    double sum = 0.0;
    weights.iterate([&sum](double& value, int i, int j) {
        sum += std::abs(value);
    });

    return sum * this->alpha_;
}
