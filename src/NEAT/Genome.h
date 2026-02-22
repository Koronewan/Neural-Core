//
// Created by korone on 1/13/25.
//

#ifndef UWU_LEARNER_GENOME_H
#define UWU_LEARNER_GENOME_H
#include <memory>
#include <unordered_map>
#include <vector>

#include "Config.h"
#include "Connection.h"
#include "InnovationCounter.h"
#include "Node.h"


class Genome
{
    std::unordered_map<int, Node> nodes_;
    std::vector<int> nodeOrder_;
    std::vector<Node> initialNodes_;
    std::vector<Connection> connections_;
    std::vector<int> outputNodesIndexes_;
    int nodeIdCounter_ = 0;
    double fitness_ = 0.0;

    const Config& config_;
    InnovationCounter& innovationCounter_;
public:
    Genome(const Config& config, InnovationCounter& innovationCounter, int inputs, int outputs);
    Genome(const Config& config, InnovationCounter& innovationCounter):
        Genome(config, innovationCounter, 0, 0) {};
    Genome(const Genome& other) = default;
    Genome &operator=(const Genome& other);
    std::vector<double> forwardPass(const std::vector<double> &input);
    void addConnection(int sourceNode, int targetNode);

    double calculateDistance(const Genome& other, double c1, double c2, double c3);
    [[nodiscard]] double getFitness() const { return fitness_; }
    void mutate();
    void addConnection();
    void addNode();
    void setFitness(double fitness) { fitness_ = fitness; }

    ~Genome() = default;
    static std::unique_ptr<Genome> crossover(const Genome &father, const Genome &mother);

    void save(const std::string &filename) const;
    void load(const std::string &filename);
};



#endif //UWU_LEARNER_GENOME_H
