//
// Created by korone on 1/10/25.
//

#ifndef MEANSQUARREDERROR_H
#define MEANSQUARREDERROR_H
#include "InterfaceLossFunction.h"
#include "MathUtils/Vector.h"

class MeanSquarredError final: public InterfaceLossFunction
{
public:
    uwu::Vector gradient(const uwu::Vector &item, const uwu::Vector &expectedItem) override;
};

#endif //MEANSQUARREDERROR_H
