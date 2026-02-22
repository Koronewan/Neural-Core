//
// Created by korone on 1/9/25.
//

#include "GlorotInitializer.h"
#include <random>

void GlorotInitializer::initialize(Matrix &weights, uwu::Vector &bias)
{
    const int nInputs = static_cast<int>(weights[0].size());
    const int nOutputs = static_cast<int>(bias.size());

    bias.fill(0.0);

    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution distribution(-std::sqrt(6.0 / (nInputs + nOutputs)),
        std::sqrt(6.0 / (nInputs + nOutputs)));

    weights.fill([&]() { return distribution(generator); });
}

GlorotInitializer *GlorotInitializer::clone() const {
    return new GlorotInitializer(*this);
}

