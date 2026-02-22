//
// Created by korone on 1/11/25.
//

#include "R2.h"
#include <cmath>

double R2::compute(const Matrix& predicted, const Matrix& actual)
{
    double r2Global = 0.0;

    for (int j = 0; j < predicted[0].size(); j++)
    {
        double ssr = 0.0;
        double actualMean = 0.0;

        for (int i = 0; i < predicted.rows(); i++)
        {
            actualMean += actual[i][j];
            ssr += std::pow(actual[i][j] - predicted[i][j], 2.0);
        }

        actualMean /= static_cast<double>(actual.rows());

        double sst = 0.0;
        for (int i = 0; i < actual.rows(); i++)
        {
            sst += std::pow(actual[i][j] - actualMean, 2.0);
        }

        r2Global += 1.0 - ssr / sst;
    }

    return r2Global / static_cast<double>(predicted[0].size());
}
