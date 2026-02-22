//
// Created by korone on 12/21/24.
//

#ifndef UWU_LEARNER_INTERFACEACTIVATIONFUNCTION_H
#define UWU_LEARNER_INTERFACEACTIVATIONFUNCTION_H
#include "MathUtils/Vector.h"
#include <string>

class InterfaceActivationFunction
{
public:
    virtual void activate(uwu::Vector &output) = 0;
    virtual void derivative(uwu::Vector &output) = 0;
    [[nodiscard]] virtual InterfaceActivationFunction* clone() const = 0;
    virtual ~InterfaceActivationFunction() = default;
    [[nodiscard]] virtual std::string getType() const = 0;
};

#endif //UWU_LEARNER_INTERFACEACTIVATIONFUNCTION_H
