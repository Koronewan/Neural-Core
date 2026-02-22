//
// Created by korone on 12/21/24.
//

#ifndef UWU_LEARNER_INTERFACELOSSFUNCTION_H
#define UWU_LEARNER_INTERFACELOSSFUNCTION_H
#include <vector>
#include "MathUtils/Vector.h"

class InterfaceLossFunction
{
public:
    virtual uwu::Vector gradient(const uwu::Vector& item, const uwu::Vector& expectedItem) = 0;

    virtual ~InterfaceLossFunction() = default;
};

#endif //UWU_LEARNER_INTERFACELOSSFUNCTION_H
