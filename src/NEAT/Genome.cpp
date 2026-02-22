//
// Created by korone on 1/13/25.
//

#include "Genome.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <unordered_set>
#include <execution>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "GeneticUtils.h"
#include "TopologicalSort.h"

Genome::Genome(const Config &config, InnovationCounter &innovationCounter,
               const int inputs, const int outputs): config_(config), innovationCounter_(innovationCounter)
{
    for (int i = 0; i < inputs; i++)
    {
        this->initialNodes_.emplace_back(NodeType::Sensor, i);
        this->nodeIdCounter_++;
    }

    for (int i = 0; i < outputs; i++)
    {
        this->nodes_[inputs + i] = Node(Output, inputs + i);
        this->outputNodesIndexes_.push_back(i + inputs);
        this->nodeIdCounter_++;
    }

    for (int i = 0; i < inputs; i++)
        {
        for (int j = 0; j < outputs; j++)
            {
            double weight = GeneticUtils::randomDouble(-config_.weightRange, config_.weightRange);
            this->connections_.emplace_back(i, inputs + j,
                weight, this->innovationCounter_.getNextInnovation());
        }
    }
}

Genome & Genome::operator=(const Genome &other)
{
    nodes_ = other.nodes_;
    connections_ = other.connections_;
    outputNodesIndexes_ = other.outputNodesIndexes_;
    initialNodes_ = other.initialNodes_;
    nodeIdCounter_ = other.nodeIdCounter_;
    fitness_ = other.fitness_;

    return *this;
}

std::vector<double> Genome::forwardPass(const std::vector<double> &input)
{
    std::vector<double> output;
    output.reserve(this->outputNodesIndexes_.size());

    for (auto &node : this->initialNodes_)
    {
        node.value = input[node.id];
    }

    std::unordered_map<int, std::vector<const Connection*>> outputConnections;
    for (const auto &conn : this->connections_)
    {
        outputConnections[conn.outputNode].push_back(&conn);
    }

    for (auto &[id, node] : this->nodes_)
    {
        auto &[type, value, bias, activation, identifier] = node;

        double sum = bias;
        if (outputConnections.find(identifier) != outputConnections.end())
        {
            for (const auto conn : outputConnections[identifier])
            {
                if (!conn->enabled) continue;
                double inputValue = (conn->inputNode < input.size())
                                        ? this->initialNodes_[conn->inputNode].value
                                        : this->nodes_.at(conn->inputNode).value;
                sum += inputValue * conn->weight;
            }
        }

        value = GeneticUtils::geneticActivation(sum, activation);
    }

    for (int outputNodeId : this->outputNodesIndexes_)
    {
        output.push_back(this->nodes_[outputNodeId].value);
    }

    return output;
}

bool dfs(const int current, const int target, const std::unordered_map<int,
             std::vector<int> > &adjacencyList, std::unordered_set<int> &visited)
{
    if (current == target) return true;
    if (visited.count(current)) return false;

    visited.insert(current);

    if (adjacencyList.count(current))
    {
        for (int neighbor : adjacencyList.at(current))
        {
            if (dfs(neighbor, target, adjacencyList, visited))
            {
                return true;
            }
        }
    }
    return false;
}

bool wouldCreateCycle(const int source, const int target, const std::vector<Connection>& connections)
{
    std::unordered_map<int, std::vector<int>> adjacencyList;

    for (const auto& conn : connections)
        {
        if (conn.enabled)
            {
            adjacencyList[conn.inputNode].push_back(conn.outputNode);
        }
    }

    // Perform DFS to check reachability
    std::unordered_set<int> visited;
    return dfs(target, source, adjacencyList, visited);
}

void Genome::addConnection(int sourceNode, int targetNode)
{
    if (wouldCreateCycle(sourceNode, targetNode, this->connections_))
    {
        return;
    }

    // Add the connection if no cycle is created
    this->connections_.emplace_back(sourceNode, targetNode,
        GeneticUtils::randomDouble(-config_.weightRange, config_.weightRange),
        this->innovationCounter_.getNextInnovation());
}


double Genome::calculateDistance(const Genome &other, const double c1, const double c2, const double c3)
{
    double weightDelta = 0.0;
    int matchingWeights = 0;
    int excess = 0;
    int disjoint = 0;

    auto it1 = this->connections_.begin();
    auto it2 = other.connections_.begin();

    while (it1 != this->connections_.end() && it2 != other.connections_.end())
    {
        if (it1->innovation == it2->innovation)
        {
            weightDelta += std::abs(it1->weight - it2->weight);
            matchingWeights++;
            ++it1;
            ++it2;
        }
        else if (it1->innovation < it2->innovation)
        {
            ++it1;
            disjoint++;
        }
        else
        {
            ++it2;
            disjoint++;
        }
    }

    excess += static_cast<int>(std::distance(it1, this->connections_.end()));
    excess += static_cast<int>(std::distance(it2, other.connections_.end()));

    const int maxGenes = static_cast<int>(std::max(this->connections_.size(), other.connections_.size()));
    const double N = (maxGenes < this->config_.minNormalizerThreshold) ? 1.0: maxGenes;
    return c1 * excess / N + c2 * disjoint / N + c3 * (matchingWeights > 0 ? weightDelta / matchingWeights : 0.0);
}

void Genome::mutate()
{
    for (auto& connection: this->connections_)
    {
        if (!connection.enabled)
            continue;

        if (GeneticUtils::randomChance(this->config_.weightReplacementRate))
        {
            connection.weight = GeneticUtils::randomDouble(-config_.weightRange, config_.weightRange);
        }
        else if (GeneticUtils::randomChance(this->config_.weightPerturbationRate))
        {
            connection.weight += GeneticUtils::randomDouble(-config_.weightPerturbationRange,
                config_.weightPerturbationRange);
        }
    }

    for (auto& [id, node]: this->nodes_)
    {
        if (GeneticUtils::randomChance(this->config_.activationReplacementRate))
        {
            node.activation = GeneticUtils::randomInt(0, GeneticUtils::activations);
        }
        else if (GeneticUtils::randomChance(this->config_.biasPerturbationRate))
        {
            node.bias = GeneticUtils::randomDouble(-config_.biasPerturbationRange, config_.biasPerturbationRange);
        }
    }

    if (GeneticUtils::randomChance(this->config_.addConnectionRate))
    {
        this->addConnection();
    }

    if (GeneticUtils::randomChance(this->config_.addNodeRate))
    {
        this->addNode();
    }

    this->nodeOrder_ = TopologicalSort::topologicalSort(this->nodes_, this->connections_);
}

void Genome::addConnection()
{
    const int totalNodes = static_cast<int>(this->initialNodes_.size() + this->nodes_.size());

    const int inputNode = GeneticUtils::randomInt(
        static_cast<int>(this->initialNodes_.size()),
        totalNodes - 1
    );

    const int outputNode = GeneticUtils::randomInt(
        static_cast<int>(this->initialNodes_.size()),
        std::max(static_cast<int>(this->initialNodes_.size()), totalNodes - 1)
    );

    if (inputNode == outputNode) {
        return;
    }

    for (const auto connection: this->connections_)
    {
        if (connection.inputNode == inputNode && connection.outputNode == outputNode)
            return;
    }

    this->addConnection(inputNode, outputNode);
}

void Genome::addNode()
{
    // Select a random connection
    Connection &conn = GeneticUtils::randomElement(this->connections_);
    conn.enabled = false;

    // Store input and output nodes to avoid invalidation
    int inputNode = conn.inputNode;
    int outputNode = conn.outputNode;

    // Add a new hidden node
    this->nodes_[nodeIdCounter_] = Node(Hidden, this->nodeIdCounter_);

    // Add new connections using the stored node IDs
    double weight = GeneticUtils::randomDouble(-config_.weightRange, config_.weightRange);
    this->connections_.emplace_back(inputNode, this->nodeIdCounter_, weight, this->innovationCounter_.getNextInnovation());

    weight = GeneticUtils::randomDouble(-config_.weightRange, config_.weightRange);
    this->connections_.emplace_back(this->nodeIdCounter_, outputNode, weight, this->innovationCounter_.getNextInnovation());

    // Increment the node ID counter
    this->nodeIdCounter_++;
}


void addNodeIfMissing(std::unordered_map<int, Node>& nodes, const std::unordered_map<int, Node>& sourceNodes,
    const int nodeId)
{
    if (nodes.find(nodeId) != nodes.end())
    {
        return;
    }

    nodes[nodeId] = sourceNodes.at(nodeId);
}

std::unique_ptr<Genome> Genome::crossover(const Genome& father, const Genome& mother)
{
    auto child = std::make_unique<Genome>(father.config_, father.innovationCounter_);

    const Genome* fitterParent = &father;
    const Genome* otherParent = &mother;

    if (father.fitness_ < mother.fitness_)
    {
        fitterParent = &mother;
        otherParent = &father;
    }

    child->initialNodes_ = fitterParent->initialNodes_;
    child->nodeIdCounter_ = fitterParent->nodeIdCounter_;
    child->outputNodesIndexes_ = fitterParent->outputNodesIndexes_;
    auto it1 = fitterParent->connections_.begin();
    auto it2 = otherParent->connections_.begin();

    while (it1 != fitterParent->connections_.end() && it2 != otherParent->connections_.end())
    {
        if (it1->innovation == it2->innovation)
        {
            if (GeneticUtils::randomChance(0.5))
            {
                auto selectedConn = *it1;
                child->connections_.push_back(selectedConn);

                if (selectedConn.inputNode < child->initialNodes_.size())
                {
                    addNodeIfMissing(child->nodes_, fitterParent->nodes_, selectedConn.outputNode);
                    ++it1;
                    ++it2;
                    continue;
                }

                addNodeIfMissing(child->nodes_, fitterParent->nodes_, selectedConn.inputNode);
                addNodeIfMissing(child->nodes_, fitterParent->nodes_, selectedConn.outputNode);
            }
            else
            {
                auto selectedConn = *it2;
                child->connections_.push_back(selectedConn);

                if (selectedConn.inputNode < child->initialNodes_.size())
                {
                    addNodeIfMissing(child->nodes_, otherParent->nodes_, selectedConn.outputNode);
                    ++it1;
                    ++it2;
                    continue;
                }

                addNodeIfMissing(child->nodes_, otherParent->nodes_, selectedConn.inputNode);
                addNodeIfMissing(child->nodes_, otherParent->nodes_, selectedConn.outputNode);
            }

            ++it1;
            ++it2;
        }
        else if (it1->innovation < it2->innovation)
        {
            child->connections_.push_back(*it1);

            if (it1->inputNode < child->initialNodes_.size())
            {
                addNodeIfMissing(child->nodes_, fitterParent->nodes_, it1->outputNode);
                ++it1;
                continue;
            }

            addNodeIfMissing(child->nodes_, fitterParent->nodes_, it1->inputNode);
            addNodeIfMissing(child->nodes_, fitterParent->nodes_, it1->outputNode);

            ++it1;
        }
        else
        {
            ++it2;
        }
    }

    while (it1 != fitterParent->connections_.end())
    {
        child->connections_.push_back(*it1);

        if (it1->inputNode < child->initialNodes_.size())
        {
            addNodeIfMissing(child->nodes_, fitterParent->nodes_, it1->outputNode);
            ++it1;
            continue;
        }

        addNodeIfMissing(child->nodes_, fitterParent->nodes_, it1->inputNode);
        addNodeIfMissing(child->nodes_, fitterParent->nodes_, it1->outputNode);

        ++it1;
    }

    return child;
}

void Genome::save(const std::string &fileName) const {
    std::ofstream ofs(fileName, std::ios::binary);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to open file for saving genome.");
    }

    // Guardar nodos
    size_t numNodes = nodes_.size();
    ofs.write(reinterpret_cast<const char*>(&numNodes), sizeof(numNodes));
    for (const auto& [id, node] : nodes_) {
        ofs.write(reinterpret_cast<const char*>(&id), sizeof(id));
        ofs.write(reinterpret_cast<const char*>(&node.type), sizeof(node.type));
        ofs.write(reinterpret_cast<const char*>(&node.value), sizeof(node.value));
        ofs.write(reinterpret_cast<const char*>(&node.bias), sizeof(node.bias));
        ofs.write(reinterpret_cast<const char*>(&node.activation), sizeof(node.activation));
        ofs.write(reinterpret_cast<const char*>(&node.id), sizeof(node.id));
    }

    // Guardar el orden de los nodos
    size_t orderSize = nodeOrder_.size();
    ofs.write(reinterpret_cast<const char*>(&orderSize), sizeof(orderSize));
    ofs.write(reinterpret_cast<const char*>(nodeOrder_.data()), orderSize * sizeof(int));

    // Guardar initialNodes_
    size_t numInitialNodes = initialNodes_.size();
    ofs.write(reinterpret_cast<const char*>(&numInitialNodes), sizeof(numInitialNodes));
    for (const auto& node : initialNodes_) {
        ofs.write(reinterpret_cast<const char*>(&node.type), sizeof(node.type));
        ofs.write(reinterpret_cast<const char*>(&node.value), sizeof(node.value));
        ofs.write(reinterpret_cast<const char*>(&node.bias), sizeof(node.bias));
        ofs.write(reinterpret_cast<const char*>(&node.activation), sizeof(node.activation));
        ofs.write(reinterpret_cast<const char*>(&node.id), sizeof(node.id));
    }

    // Guardar conexiones
    size_t numConnections = connections_.size();
    ofs.write(reinterpret_cast<const char*>(&numConnections), sizeof(numConnections));
    for (const auto& connection : connections_) {
        ofs.write(reinterpret_cast<const char*>(&connection.inputNode), sizeof(connection.inputNode));
        ofs.write(reinterpret_cast<const char*>(&connection.outputNode), sizeof(connection.outputNode));
        ofs.write(reinterpret_cast<const char*>(&connection.weight), sizeof(connection.weight));
        ofs.write(reinterpret_cast<const char*>(&connection.enabled), sizeof(connection.enabled));
        ofs.write(reinterpret_cast<const char*>(&connection.innovation), sizeof(connection.innovation));
    }

    // Guardar índices de nodos de salida
    size_t outputSize = outputNodesIndexes_.size();
    ofs.write(reinterpret_cast<const char*>(&outputSize), sizeof(outputSize));
    ofs.write(reinterpret_cast<const char*>(outputNodesIndexes_.data()), outputSize * sizeof(int));

    // guardar node id counter
    ofs.write(reinterpret_cast<const char*>(&nodeIdCounter_), sizeof(nodeIdCounter_));

    // Guardar fitness
    ofs.write(reinterpret_cast<const char*>(&fitness_), sizeof(fitness_));
}

void Genome::load(const std::string &fileName) {
    std::ifstream ifs(fileName, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open file for loading genome.");
    }

    // Limpiar datos actuales
    nodes_.clear();
    connections_.clear();
    nodeOrder_.clear();
    initialNodes_.clear();
    outputNodesIndexes_.clear();

    // Cargar nodos
    size_t numNodes;
    ifs.read(reinterpret_cast<char*>(&numNodes), sizeof(numNodes));
    for (size_t i = 0; i < numNodes; ++i) {
        int id;
        Node node;
        ifs.read(reinterpret_cast<char*>(&id), sizeof(id));
        ifs.read(reinterpret_cast<char*>(&node.type), sizeof(node.type));
        ifs.read(reinterpret_cast<char*>(&node.value), sizeof(node.value));
        ifs.read(reinterpret_cast<char*>(&node.bias), sizeof(node.bias));
        ifs.read(reinterpret_cast<char*>(&node.activation), sizeof(node.activation));
        ifs.read(reinterpret_cast<char*>(&node.id), sizeof(node.id));
        nodes_[id] = node;
    }

    // Cargar el orden de los nodos
    size_t orderSize;
    ifs.read(reinterpret_cast<char*>(&orderSize), sizeof(orderSize));
    nodeOrder_.resize(orderSize);
    ifs.read(reinterpret_cast<char*>(nodeOrder_.data()), orderSize * sizeof(int));

    // Cargar initialNodes_
    size_t numInitialNodes;
    ifs.read(reinterpret_cast<char*>(&numInitialNodes), sizeof(numInitialNodes));
    for (size_t i = 0; i < numInitialNodes; ++i) {
        Node node;
        ifs.read(reinterpret_cast<char*>(&node.type), sizeof(node.type));
        ifs.read(reinterpret_cast<char*>(&node.value), sizeof(node.value));
        ifs.read(reinterpret_cast<char*>(&node.bias), sizeof(node.bias));
        ifs.read(reinterpret_cast<char*>(&node.activation), sizeof(node.activation));
        ifs.read(reinterpret_cast<char*>(&node.id), sizeof(node.id));
        initialNodes_.emplace_back(node);
    }

    // Cargar conexiones
    size_t numConnections;
    ifs.read(reinterpret_cast<char*>(&numConnections), sizeof(numConnections));
    for (size_t i = 0; i < numConnections; ++i) {
        Connection connection(0, 0, 0.0, 0);
        ifs.read(reinterpret_cast<char*>(&connection.inputNode), sizeof(connection.inputNode));
        ifs.read(reinterpret_cast<char*>(&connection.outputNode), sizeof(connection.outputNode));
        ifs.read(reinterpret_cast<char*>(&connection.weight), sizeof(connection.weight));
        ifs.read(reinterpret_cast<char*>(&connection.enabled), sizeof(connection.enabled));
        ifs.read(reinterpret_cast<char*>(&connection.innovation), sizeof(connection.innovation));
        connections_.push_back(connection);
    }

    // Cargar índices de nodos de salida
    size_t outputSize;
    ifs.read(reinterpret_cast<char*>(&outputSize), sizeof(outputSize));
    outputNodesIndexes_.resize(outputSize);
    ifs.read(reinterpret_cast<char*>(outputNodesIndexes_.data()), outputSize * sizeof(int));

    // Cargar node id counter
    ifs.read(reinterpret_cast<char*>(&nodeIdCounter_), sizeof(nodeIdCounter_));

    // Cargar fitness
    ifs.read(reinterpret_cast<char*>(&fitness_), sizeof(fitness_));
}



