//
// Created by korone on 1/9/25.
//

#include "HeInitializer.h"
#include <random>

void HeInitializer::initialize(Matrix &weights, uwu::Vector &bias)
{
    const int nInputs = static_cast<int>(weights[0].size());
    bias.fill(0.0); // Rellenamos el vector de sesgos con ceros.

    std::random_device rd;
    std::mt19937 generator(rd());

    std::uniform_real_distribution distribution(-std::sqrt(6.0 / (nInputs)),
        std::sqrt(6.0 / (nInputs)));
    weights.fill([&]() { return distribution(generator); });

}

HeInitializer *HeInitializer::clone() const {
    return new HeInitializer(*this);
}
