//
// Created by korone on 12/21/24.
//

#ifndef UWU_LEARNER_INTERFACEINITIALIZER_H
#define UWU_LEARNER_INTERFACEINITIALIZER_H

#include "MathUtils/Matrix.h"

class InterfaceInitializer
{
public:
    virtual void initialize(Matrix &weights, uwu::Vector &bias) = 0;
    [[nodiscard]] virtual InterfaceInitializer* clone() const = 0;
    virtual ~InterfaceInitializer() = default;
};

#endif //UWU_LEARNER_INTERFACEINITIALIZER_H
