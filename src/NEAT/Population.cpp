//
// Created by korone on 1/13/25.
//

#include "Population.h"

#include <algorithm>
#include <iostream>

#include "GeneticUtils.h"

void Population::generateSpecies()
{
    for (auto& species : species_)
    {
        species->clear();
    }

    for (auto& genome : this->genomes_)
    {
        bool assigned = false;

        for (const auto& specie : this->species_)
        {
            if (const double distance = genome->calculateDistance(specie->getRepresentative(),
                this->config_.c1, this->config_.c2, this->config_.c3); distance < this->deltaThreshold_)
            {
                specie->addGenome(std::move(genome));
                assigned = true;
                break;
            }
        }

        if (!assigned)
        {
            auto specie = std::make_unique<Specie>(this->config_, std::move(genome));
            this->species_.push_back(std::move(specie));
        }
    }

    this->species_.erase(
        std::remove_if(
            this->species_.begin(),
            this->species_.end(),
            [](const auto& specie) {
                return specie->genomes() == 0;
            }),
        this->species_.end());


    // Dynamic adjustment of deltaThreshold
    if (this->species_.size() > this->config_.targetSpeciesCounter)
    {
        this->deltaThreshold_ += this->config_.deltaThresholdAdjustment;
    }
    else
    {
        this->deltaThreshold_ -= this->config_.deltaThresholdAdjustment;
    }

}

void Population::calculateAdjustedFitnessAndOffsprings()
{
    double globalFitness = 0.0;
    for (const auto& specie : this->species_)
    {
        globalFitness += specie->calculateAdjustedFitness();
    }

    int totalOffspring = 0;
    for (const auto& specie : this->species_)
    {
        totalOffspring += specie->calculateOffspring(static_cast<int>(this->genomes_.size()), globalFitness);
    }

    for (const auto& specie : this->species_)
    {
        specie->setOffspring(std::max(this->config_.minSpecieSize, specie->getOffspring())
            * this->config_.populationSize / totalOffspring);
    }
}

void Population::generateNextPopulation(int input, int output)
{
    std::vector<std::unique_ptr<Genome>> nextPopulation;
    for (const auto& specie : this->species_)
    {
        std::vector<std::unique_ptr<Genome>> nextSpeciePopulation = specie->generateOffsprings();
        nextPopulation.insert(nextPopulation.end(),
            std::make_move_iterator(nextSpeciePopulation.begin()),
            std::make_move_iterator(nextSpeciePopulation.end()));
    }

    this->genomes_ = std::move(nextPopulation);
}

std::vector<Genome> Population::getGenomes() const {
    std::vector<Genome> genomes;

    for (auto& genome : this->genomes_)
    {
        genomes.push_back(*genome);
    }

    return genomes;
}

void Population::setFitness(const std::vector<double> &fitness)
{
    for (int i = 0; i < fitness.size(); i++)
    {
        this->genomes_[i]->setFitness(fitness[i]);

        if (fitness[i] > this->bestFit_)
        {
            this->bestGenome_ = std::make_unique<Genome>(*this->genomes_[i]);
            this->bestFit_= fitness[i];
        }
    }
}

void Population::addRandomGenom(int input, int output)
{
    this->genomes_.push_back(std::make_unique<Genome>(this->config_,
        this->innovationCounter_, input, output));
}

