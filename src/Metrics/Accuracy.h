//
// Created by aapr6 on 1/12/25.
//

#ifndef ACCURACY_H
#define ACCURACY_H
#include "InterfaceMetric.h"

class Accuracy final: public InterfaceMetric
{
public:
    double compute(const Matrix& predicted,
        const Matrix& actual) override;
};


#endif //ACCURACY_H
