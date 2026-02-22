//
// Created by korone on 1/13/25.
//

#ifndef UWU_LEARNER_POPULATION_H
#define UWU_LEARNER_POPULATION_H
#include <utility>
#include <vector>

#include "Config.h"
#include "Genome.h"
#include "InnovationCounter.h"
#include "Specie.h"


class Population
{
    std::vector<std::unique_ptr<Genome>> genomes_;
    std::vector<std::unique_ptr<Specie>> species_;

    const Config& config_;
    InnovationCounter& innovationCounter_;

    double deltaThreshold_ = config_.deltaThreshold_;
    std::unique_ptr<Genome> bestGenome_{};
    double bestFit_ = 0.0;

public:
    Population(const Config& config, InnovationCounter& innovationCounter)
        : config_(config), innovationCounter_(innovationCounter) {}
    ~Population() = default;
    void setGenomes(std::vector<std::unique_ptr<Genome>> genoms) { this->genomes_ = std::move(genoms); };
    void generateSpecies();
    void calculateAdjustedFitnessAndOffsprings();
    void generateNextPopulation(int input, int output);

    [[nodiscard]] double getBestFitness() const { return bestGenome_->getFitness(); }
    [[nodiscard]] Genome& getBestGenome() const { return *bestGenome_; }
    int getPopulationSize() const { return genomes_.size(); }
    int getSpeciesSize() const { return species_.size(); }
    std::vector<Genome> getGenomes() const;
    void setFitness(const std::vector<double> &fitness);
    void addRandomGenom(int input, int output);
};

#endif //UWU_LEARNER_POPULATION_H
