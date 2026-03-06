//
// Created by korone on 12/21/24.
//

#ifndef NEURAL_CORE_INTERFACEINITIALIZER_H
#define NEURAL_CORE_INTERFACEINITIALIZER_H

#include "MathUtils/Matrix.h"

class InterfaceInitializer
{
public:
    virtual void initialize(Matrix &weights, uwu::Vector &bias) = 0;
    [[nodiscard]] virtual InterfaceInitializer* clone() const = 0;
    virtual ~InterfaceInitializer() = default;
};

#endif //NEURAL_CORE_INTERFACEINITIALIZER_H
