//
// Created by korone on 12/21/24.
//

#include "NeuralNetwork.h"

#include <chrono>
#include <fstream>

#include "Layers/WeightedLayer.h"
#include "Layers/Dropout.h"

void NeuralNetwork::compile(InterfaceOptimizer *optimizer, InterfaceLossFunction *loss, InterfaceMetric * &metric)
{
    this->metric_ = metric;
    this->optimizer_ = OptimizerManager(optimizer, this->layers_);
    this->loss_ = loss;
}

void NeuralNetwork::addLayer(InterfaceLayer *layer)
{
    this->layers_.push_back(layer);
    this->eventManager_.subscribe(layer, "BatchStart");
    this->eventManager_.subscribe(layer, "BatchEnd");
}

void NeuralNetwork::fit(const DataSet &dataSet, const int epochs, const int batchSize, const double validationSplit,
    EarlyStopping earlyStopping)
{
    this->eventManager_.notify("FitStart");

    auto[trainingDataSet, validationDataSet] = dataSet.split(validationSplit);
    const Matrix trainingFeatures = trainingDataSet.getFeatures();
    const Matrix trainingLabels = trainingDataSet.getLabels();
    const Matrix validationFeatures = validationDataSet.getFeatures();
    const Matrix validationLabels = validationDataSet.getLabels();
    const int num_batches = std::round(static_cast<double>(trainingDataSet.getItems()) / batchSize) * epochs;
    std::cout << "Starting Fit" << std::endl;

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        auto epochStart = std::chrono::high_resolution_clock::now();
        this->eventManager_.notify("EpochStart");
        for (int i = 0; i < num_batches; ++i)
        {
            this->eventManager_.notify("BatchStart");
            for (int j = 0; j < batchSize; j++)
            {
                // This is the forward pass
                Matrix activations;
                auto activation = uwu::Vector(trainingFeatures[i * batchSize + j]);
                activations.push_back(activation);

                for (const auto layer : this->layers_)
                {
                    activation = layer->forward(activation);
                    activations.push_back(activation);
                }

                // End of forward pass, we have all the z values, and it's corresponding activated values

                // Start of backpropagation
                uwu::Vector error = this->loss_->gradient(activation, uwu::Vector(trainingLabels[i * batchSize + j]));
                for (int k = this->layers_.size() - 1; k >= 0; k--)
                {
                    this->layers_[k]->backward(error, uwu::Vector(activations[k]));
                }
            }

            for (const auto layer : this->layers_)
            {
                if (const auto weightedLayer = dynamic_cast<WeightedLayer*>(layer))
                {
                    this->optimizer_.update(weightedLayer);
                }
            }

            this->eventManager_.notify("BatchEnd");
        }

        this->eventManager_.notify("EpochEnd");
        Matrix output = this->predict(trainingFeatures);

        auto epochEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> epochDuration = epochEnd - epochStart;
        std::cout << "Epoch " << epoch + 1 << " duration: " << epochDuration.count() << " seconds\n";
        double metric = this->metric_->compute(output, trainingLabels);
        std::cout << "Training Metric: " << metric
            << "\n";

        if (validationSplit > 0)
        {
            Matrix valPredictions = this->predict(validationFeatures);
            metric = this->metric_->compute(valPredictions, validationLabels);
            std::cout << "Validation Metric: " << metric
                << "\n";
        }

        earlyStopping.evaluate(metric);
        if (earlyStopping.shouldStop()) {
            std::cout << "Early Stopping" << std::endl;
            return;
        }
    }
}

Matrix NeuralNetwork::predict(const Matrix &input)
{
    Matrix predictions;

    for (int i = 0; i < input.rows(); ++i) {
        uwu::Vector activation = input[i];
        for (const auto& layer : this->layers_)
        {
            activation = layer->forward(activation);
        }
        predictions.push_back(activation);
    }

    return predictions;
}

std::string NeuralNetwork::getLayersInfo() const {
    std::ostringstream oss;

    oss << "Neural Network Information:\n";
    oss << "Total Layers: " << layers_.size() << "\n\n";

    for (size_t i = 0; i < layers_.size(); ++i) {
        oss << "Layer " << i + 1 << ":\n";
        oss << layers_[i]->getInfo() << "\n"; // Llama a getInfo de cada capa
    }

    return oss.str();
}
