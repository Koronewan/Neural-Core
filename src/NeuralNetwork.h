//
// Created by korone on 12/21/24.
//

#ifndef UWU_LEARNER_NEURALNETWORK_H
#define UWU_LEARNER_NEURALNETWORK_H

#include <vector>

#include "DataSet.h"
#include "EarlyStopping.h"
#include "Layers/InterfaceLayer.h"
#include "Optimizers/InterfaceOptimizer.h"
#include "Loss/InterfaceLossFunction.h"
#include "Metrics/InterfaceMetric.h"
#include "Events/EventManager.h"
#include "Optimizers/OptimizerManager.h"

class NeuralNetwork
{
    std::vector<InterfaceLayer*> layers_;
    InterfaceMetric* metric_{};
    OptimizerManager optimizer_;
    InterfaceLossFunction* loss_{};
    EventManager eventManager_;

public:
    NeuralNetwork() = default;
    void compile(InterfaceOptimizer *optimizer, InterfaceLossFunction *loss, InterfaceMetric* &metrics);
    void addLayer(InterfaceLayer* layer);
    void fit(const DataSet &dataSet, int epochs, int batchSize, double validationSplit,
        EarlyStopping earlyStopping = EarlyStopping());

    Matrix predict(const Matrix &input);

    void saveBinary(const std::string& filePath) const;
    void loadBinary(const std::string& filePath);
    std::string getLayersInfo() const;
};

#endif //UWU_LEARNER_NEURALNETWORK_H
