//
// Created by korone on 1/13/25.
//

#ifndef UWU_LEARNER_CONFIG_H
#define UWU_LEARNER_CONFIG_H

class Config
{
public:
    // Population Config Parameters
    int targetSpeciesCounter = 20;
    double deltaThreshold_ = 3.0;
    double deltaThresholdAdjustment = 0.1;
    double c1 = 1.0;
    double c2 = 1.0;
    double c3 = 0.4;

    // Genome Config Parameters
    int minNormalizerThreshold = 20;
    double weightRange = 1.0;
    double weightPerturbationRange = 0.1;

    // Mutation Parameters
    double weightPerturbationRate = 0.8; // Chance to perturb weights
    double weightReplacementRate = 0.1; // Chance to replace weights
    double addConnectionRate = 0.2; // Chance to add a connection
    double addNodeRate = 0.2; // Chance to add a node
    double activationReplacementRate = 0.05;
    double biasPerturbationRate = 0.8;
    double biasPerturbationRange = 0.1;

    // Crossover Parameters
    double crossoverRate = 0.75; // Chance to perform crossover
    double elitePreservationRate = 0.1; // Percentage of elites preserved

    // Population Parameters
    int populationSize = 100;
    int minSpecieSize = 5;


    Config() = default;
    ~Config() = default;
};

#endif //UWU_LEARNER_CONFIG_H
