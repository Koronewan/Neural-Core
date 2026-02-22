//
// Created by aapr6 on 1/12/25.
//

#include "F1Score.h"
#include <cmath>

double F1Score::compute(const Matrix& predicted, const Matrix& actual)
{
    double truePositives = 0.0;
    double falsePositives = 0.0;
    double falseNegatives = 0.0;

    for (int i = 0; i < predicted.rows(); ++i)
    {
        for (int j = 0; j < predicted[i].size(); ++j)
        {
            double predictedValue = std::round(predicted[i][j]);
            double actualValue = actual[i][j];

            if (predictedValue == 1.0 && actualValue == 1.0)
            {
                truePositives += 1.0; // TP
            }
            else if (predictedValue == 1.0 && actualValue == 0.0)
            {
                falsePositives += 1.0; // FP
            }
            else if (predictedValue == 0.0 && actualValue == 1.0)
            {
                falseNegatives += 1.0; // FN
            }
        }
    }

    // Verificar si precision y recall pueden calcularse exactamente
    if (truePositives == 0.0)
    {
        return 0.0; // Sin positivos verdaderos, F1 es 0
    }

    double precision = truePositives / (truePositives + falsePositives);
    double recall = truePositives / (truePositives + falseNegatives);

    // Si ambas son exactamente 1, devolver 1 directamente
    if (precision == 1.0 && recall == 1.0)
    {
        return 1.0;
    }

    // Calcular el F1 Score estándar
    return 2.0 * (precision * recall) / (precision + recall);
}