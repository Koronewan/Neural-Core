//
// Created by aapr6 on 1/12/25.
//

#ifndef F1SCORE_H
#define F1SCORE_H
#include "InterfaceMetric.h"

class F1Score final : public InterfaceMetric
{
public:
    double compute(const Matrix& predicted, const Matrix& actual) override;
};

#endif //F1SCORE_H
