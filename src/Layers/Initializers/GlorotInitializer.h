//
// Created by korone on 1/9/25.
//

#ifndef NEURAL_CORE_GLOROTINITIALIZER_H
#define NEURAL_CORE_GLOROTINITIALIZER_H
#include "InterfaceInitializer.h"
#include <vector>

class GlorotInitializer final: public InterfaceInitializer
{
public:
    void initialize(Matrix &weights, uwu::Vector &bias) override;
    [[nodiscard]] GlorotInitializer* clone() const override;
};

#endif //NEURAL_CORE_GLOROTINITIALIZER_H
