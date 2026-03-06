//
// Created by korone on 1/7/25.
//

#ifndef NEURAL_CORE_OPTIMIZERMANAGER_H
#define NEURAL_CORE_OPTIMIZERMANAGER_H
#include <unordered_map>

#include "InterfaceOptimizer.h"
#include "OptimizerState.h"
#include "Layers/WeightedLayer.h"


class OptimizerManager
{
    InterfaceOptimizer* optimizer_{};
    std::unordered_map<WeightedLayer*, OptimizerState*> states_;
public:
    OptimizerManager() = default;
    OptimizerManager(InterfaceOptimizer* optimizer, std::vector<InterfaceLayer*> layers);
    ~OptimizerManager() = default;
    void update(WeightedLayer* layer);
};



#endif //NEURAL_CORE_OPTIMIZERMANAGER_H
