//
// Created by aapr6 on 1/10/25.
//

#ifndef RMSPROPSTATE_H
#define RMSPROPSTATE_H

#include <vector>
#include "../OptimizerState.h"

class RMSProp;

class RMSPropState: public OptimizerState
{
    uwu::Vector meanSquareBias;
    Matrix meanSquareWeights;

    friend class RMSProp;
};

#endif //RMSPROPSTATE_H
