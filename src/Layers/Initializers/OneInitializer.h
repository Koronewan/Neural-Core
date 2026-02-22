//
// Created by korone on 1/9/25.
//

#ifndef ONEINITIALIZER_H
#define ONEINITIALIZER_H

#include <vector>
#include "InterfaceInitializer.h"

class OneInitializer final: public InterfaceInitializer
{
public:
    void initialize(Matrix &weights, uwu::Vector &bias) override;
    [[nodiscard]] OneInitializer* clone() const override;
};

#endif //ONEINITIALIZER_H
