//
// Created by korone on 1/7/25.
//

#include "OptimizerManager.h"

#include "Adam/Adam.h"
#include "Layers/WeightedLayer.h"
#include "RMSProp/RMSProp.h"
#include "SGD/SGD.h"

OptimizerManager::OptimizerManager(InterfaceOptimizer *optimizer, const std::vector<InterfaceLayer *> layers)
{
    this->optimizer_ = optimizer;
    for (const auto layer : layers)
    {
        if (const auto weightedLayer = dynamic_cast<WeightedLayer*>(layer))
        {
            if (dynamic_cast<Adam*>(optimizer))
            {
                this->states_[weightedLayer] = new AdamState();
            }
            else if (dynamic_cast<RMSProp*>(optimizer))
            {
                this->states_[weightedLayer] = new RMSPropState();
            }
            else if (dynamic_cast<SGD*>(optimizer))
            {
                this->states_[weightedLayer] = new SGDState();
            }
        }
    }
}

void OptimizerManager::update(WeightedLayer *layer)
{
    auto& weights = layer->getWeights();
    auto weightGradients = layer->getWeightsGradients();
    auto& biases = layer->getBiases();
    auto biasGradients = layer->getBiasesGradients();

    this->optimizer_->update(biasGradients, biases, *this->states_[layer]);
    this->optimizer_->update(weightGradients, weights, *this->states_[layer]);
}
