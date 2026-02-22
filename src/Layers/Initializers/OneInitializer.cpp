//
// Created by korone on 1/9/25.
//

#include "OneInitializer.h"

void OneInitializer::initialize(Matrix &weights, uwu::Vector &bias)
{
    bias.fill(0.0);

    weights.fill(1.0);
}

OneInitializer* OneInitializer::clone() const {
    return new OneInitializer(*this);
}
