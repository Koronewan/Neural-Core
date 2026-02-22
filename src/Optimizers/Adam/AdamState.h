//
// Created by aapr6 on 1/10/25.
//

#ifndef ADAMSTATE_H
#define ADAMSTATE_H
#include <vector>

#include "../OptimizerState.h"

class Adam;

class AdamState: public OptimizerState
{
    uwu::Vector momentumBias;
    uwu::Vector velocitiesBias;
    Matrix momentumWeights;
    Matrix velocitiesWeights;
    double t = 0.0;
public:
    AdamState() = default;
    friend class Adam;
};

#endif //ADAMSTATE_H
