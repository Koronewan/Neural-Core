//
// Created by korone on 1/9/25.
//

#ifndef UWU_LEARNER_GLOROTINITIALIZER_H
#define UWU_LEARNER_GLOROTINITIALIZER_H
#include "InterfaceInitializer.h"
#include <vector>

class GlorotInitializer final: public InterfaceInitializer
{
public:
    void initialize(Matrix &weights, uwu::Vector &bias) override;
    [[nodiscard]] GlorotInitializer* clone() const override;
};

#endif //UWU_LEARNER_GLOROTINITIALIZER_H
