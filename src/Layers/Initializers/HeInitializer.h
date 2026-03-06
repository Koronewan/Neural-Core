//
// Created by korone on 1/9/25.
//

#ifndef NEURAL_CORE_HEINITIALIZER_H
#define NEURAL_CORE_HEINITIALIZER_H
#include "InterfaceInitializer.h"
#include <vector>

class HeInitializer final: public InterfaceInitializer
{
public:
    void initialize(Matrix &weights, uwu::Vector &bias) override;
    [[nodiscard]] HeInitializer* clone() const override;
};

#endif //NEURAL_CORE_HEINITIALIZER_H
