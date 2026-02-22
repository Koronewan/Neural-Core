//
// Created by korone on 1/13/25.
//

#ifndef UWU_LEARNER_SELFEVOLVINGNEURALNETWORK_H
#define UWU_LEARNER_SELFEVOLVINGNEURALNETWORK_H
#include "InnovationCounter.h"
#include "Population.h"

class SelfEvolvingNeuralNetwork
{
    std::unique_ptr<Population> population_{};
    InnovationCounter& innovationCounter_;
    int inputSize_;
    int outputSize_;
    const Config& config_;
    Genome bestGenome;

public:
    SelfEvolvingNeuralNetwork(const Config& config, InnovationCounter& innovationCounter, int inputSize, int outputSize)
        : innovationCounter_(innovationCounter), inputSize_(inputSize), outputSize_(outputSize),
        config_(config), bestGenome(Genome(config, innovationCounter))
    {
        std::vector<std::unique_ptr<Genome>> initialGenomes;
        initialGenomes.reserve(this->config_.populationSize);
        for (int i = 0; i < this->config_.populationSize; i++)
        {
            initialGenomes.push_back(std::make_unique<Genome>(this->config_, this->innovationCounter_,
                inputSize, outputSize));
        }

        this->population_ = std::make_unique<Population>(this->config_, this->innovationCounter_);
        this->population_->setGenomes(std::move(initialGenomes));
    }

    Genome evolve();
    [[nodiscard]] std::vector<Genome> evaluate() const;
    void setFitness(const std::vector<double>& fitness);

};

#endif //UWU_LEARNER_SELFEVOLVINGNEURALNETWORK_H
