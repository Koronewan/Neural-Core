//
// Created by aapr6 on 1/12/25.
//

#ifndef CROSSENTROPY_H
#define CROSSENTROPY_H
#include "InterfaceLossFunction.h"

class CrossEntropy final : public InterfaceLossFunction
{
public:
    uwu::Vector gradient(const uwu::Vector &item, const uwu::Vector &expectedItem) override;
};

#endif //CROSSENTROPY_H
