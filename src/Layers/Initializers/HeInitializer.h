//
// Created by korone on 1/9/25.
//

#ifndef UWU_LEARNER_HEINITIALIZER_H
#define UWU_LEARNER_HEINITIALIZER_H
#include "InterfaceInitializer.h"
#include <vector>

class HeInitializer final: public InterfaceInitializer
{
public:
    void initialize(Matrix &weights, uwu::Vector &bias) override;
    [[nodiscard]] HeInitializer* clone() const override;
};

#endif //UWU_LEARNER_HEINITIALIZER_H
