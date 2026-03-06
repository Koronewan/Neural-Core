//
// Created by korone on 12/21/24.
//

#ifndef NEURAL_CORE_INTERFACELOSSFUNCTION_H
#define NEURAL_CORE_INTERFACELOSSFUNCTION_H
#include <vector>
#include "MathUtils/Vector.h"

class InterfaceLossFunction
{
public:
    virtual uwu::Vector gradient(const uwu::Vector& item, const uwu::Vector& expectedItem) = 0;

    virtual ~InterfaceLossFunction() = default;
};

#endif //NEURAL_CORE_INTERFACELOSSFUNCTION_H
