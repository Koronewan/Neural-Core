//
// Created by korone on 1/13/25.
//

#include "Specie.h"

#include <algorithm>
#include <random>

#include "GeneticUtils.h"
#include "TopologicalSort.h"

void Specie::addGenome(std::unique_ptr<Genome> genome)
{
    genomes_.push_back(std::move(genome));
}


Genome Specie::getRepresentative() const
{
    return *this->representative_;
}

double Specie::calculateAdjustedFitness()
{
    this->adjustedFitness_ = 0.0;

    for (const auto& genome : this->genomes_)
    {
        adjustedFitness_ += genome->getFitness();
    }

    this->adjustedFitness_ /= static_cast<double>(this->genomes_.size());

    return this->adjustedFitness_;
}

int Specie::calculateOffspring(const int populationSize, const double globalFitness)
{
    return std::max(this->config_.minSpecieSize,
        static_cast<int>(this->adjustedFitness_ / globalFitness * static_cast<double>(populationSize)));
}

std::vector<std::unique_ptr<Genome>> Specie::selectElites()
{
    auto eliteCount = static_cast<int>(this->config_.elitePreservationRate * static_cast<double>(this->genomes_.size()));

    // Perform partial sort based on fitness
    std::partial_sort(this->genomes_.begin(),
                      this->genomes_.begin() + eliteCount,
                      this->genomes_.end(),
                      [](const std::unique_ptr<Genome>& a, const std::unique_ptr<Genome>& b) {
                          return a->getFitness() > b->getFitness();
                      });

    std::vector<std::unique_ptr<Genome>> elites;
    elites.reserve(eliteCount);
    for (int i = 0; i < eliteCount; ++i) {
        elites.push_back(std::make_unique<Genome>(*this->genomes_[i]));
    }

    return elites;
}


std::vector<std::unique_ptr<Genome>> Specie::generateOffsprings()
{
    std::vector<std::unique_ptr<Genome>> offsprings = this->selectElites();

    for (int i = offsprings.size(); i < this->offspring_; i++)
    {
        Genome& father = GeneticUtils::randomElement(this->genomes_);
        Genome& mother = GeneticUtils::randomElement(this->genomes_);
        std::unique_ptr<Genome> child;

        if (GeneticUtils::randomChance(this->config_.crossoverRate))
        {
            child = Genome::crossover(father, mother);
        }
        else
        {
            if (father.getFitness() < mother.getFitness())
            {
                child = std::make_unique<Genome>(mother);
            }
            else
            {
                child = std::make_unique<Genome>(father);
            }
        }

        child->mutate();
        offsprings.push_back(std::move(child));
    }

    return offsprings;
}
