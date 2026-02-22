//
// Created by aapr6 on 1/12/25.
//

#include "Accuracy.h"
#include <cmath>
#include <algorithm>

double Accuracy::compute(const Matrix& predicted, const Matrix& actual)
{
    double correct = 0.0;
    double total = static_cast<double>(predicted.size());

    for (size_t i = 0; i < predicted.size(); ++i)
        {
        // Find the predicted class (argmax)
        size_t predictedClass = std::distance(predicted[i].begin(),
                                              std::max_element(predicted[i].begin(), predicted[i].end()));
        // Find the actual class (argmax)
        size_t actualClass = std::distance(actual[i].begin(),
                                           std::max_element(actual[i].begin(), actual[i].end()));
        // Check if the prediction is correct
        if (predictedClass == actualClass) {
            correct += 1.0;
        }
    }

    return correct / total;
}