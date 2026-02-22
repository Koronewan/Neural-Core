//
// Created by korone on 1/13/25.
//

#include "SelfEvolvingNeuralNetwork.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <ostream>

Genome SelfEvolvingNeuralNetwork::evolve()
{
    this->population_->generateSpecies();

    this->population_->calculateAdjustedFitnessAndOffsprings();

    this->population_->generateNextPopulation(this->inputSize_, this->outputSize_);
    return this->bestGenome = this->population_->getBestGenome();
}

std::vector<Genome> SelfEvolvingNeuralNetwork::evaluate() const
{
    return this->population_->getGenomes();
}

void SelfEvolvingNeuralNetwork::setFitness(const std::vector<double> &fitness)
{
    this->population_->setFitness(fitness);
}
