//
// Created by korone on 1/13/25.
//

#include "GeneticUtils.h"

#include <algorithm>
#include <cmath>
#include <random>

double GeneticUtils::geneticActivation(const double x, const int activation)
{
    switch (activation)
    {
        case 0:
            return std::max(0.0, x);
        case 1:
            return std::tanh(x);
        case 2:
            return  1.0 / (1.0 + std::exp(-x));
        default:
            return x;
    }
}

double GeneticUtils::randomDouble(const double min, const double max)
{
    static std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(min, max);
    return dist(rng);
}

bool GeneticUtils::randomChance(const double probability)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0); // Range [0, 1)

    return dis(gen) < probability;
}

int GeneticUtils::randomInt(const int min, const int max)
{
    static std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution dist(min, max);
    return dist(rng);
}

