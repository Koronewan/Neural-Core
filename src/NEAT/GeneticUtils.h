//
// Created by korone on 1/13/25.
//

#ifndef  UWU_LEARNER_GENETICUTILS_H
#define  UWU_LEARNER_GENETICUTILS_H
#include <memory>
#include <vector>
#include <random>

#include "Connection.h"

class GeneticUtils
{
public:
    static constexpr int activations = 3;
    static double geneticActivation(double x, int activation);
    static double randomDouble(double min, double max);
    static bool randomChance(double probability);
    static int randomInt(int min, int max);
    template<class T>
    static T &randomElement(std::vector<std::unique_ptr<T>> &vector);

    template<class T>
    static Connection &randomElement(std::vector<T> &vector);

    template <typename T>
    static auto randomElement(T e1, T e2) -> T;
};

template<typename T>
T& GeneticUtils::randomElement(std::vector<std::unique_ptr<T>>& vector)
{

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, static_cast<int>(vector.size() - 1));

    return *vector[dis(gen)]; // Return a reference to the managed object
}

template<class T>
Connection &GeneticUtils::randomElement(std::vector<T> &vector)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, static_cast<int>(vector.size() - 1));

    return vector[dis(gen)];
}

template<typename T>
T GeneticUtils::randomElement(T e1, T e2)
{
    if (randomChance(0.5))
    {
        return e1;
    }

    return e2;
}

#endif // UWU_LEARNER_GENETICUTILS_H
