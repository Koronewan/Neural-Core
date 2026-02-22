//
// Created by korone on 1/13/25.
//

#ifndef UWU_LEARNER_SPECIE_H
#define UWU_LEARNER_SPECIE_H
#include <memory>
#include <vector>

#include "Genome.h"

class Specie
{
    std::vector<std::unique_ptr<Genome>> genomes_;
    std::unique_ptr<Genome> representative_;
    double adjustedFitness_ = 0.0;
    int offspring_ = 0;
    const Config& config_;

public:
    Specie(const Config& config, std::unique_ptr<Genome> representative)
        : config_(config) {
        representative_ = std::make_unique<Genome>(*representative);
        genomes_.push_back(std::move(representative));
    }
    ~Specie() = default;
    void addGenome(std::unique_ptr<Genome> genome);
    [[nodiscard]] Genome getRepresentative() const;
    [[nodiscard]] int getOffspring() const { return offspring_; }
    double calculateAdjustedFitness();
    void clear() { this->genomes_.clear();}

    int calculateOffspring(int populationSize, double globalFitness);

    std::vector<std::unique_ptr<Genome>> selectElites();
    std::vector<std::unique_ptr<Genome>> generateOffsprings();

    [[nodiscard]] int offspring() const
    {
        return offspring_;
    }

    void setOffspring(const int offspring)
    {
        offspring_ = offspring;
    }

    int genomes() const  { return genomes_.size(); }
};

#endif //UWU_LEARNER_SPECIE_H
