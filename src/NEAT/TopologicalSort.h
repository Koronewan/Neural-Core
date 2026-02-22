//
// Created by korone on 1/13/25.
//

#ifndef UWU_LEARNER_TOPOLOGICALSORT_H
#define UWU_LEARNER_OPOLOGICALSORT_H
#include <unordered_map>
#include <vector>

#include "Node.h"
#include "Connection.h"

class TopologicalSort
{
public:
    static std::vector<int> topologicalSort(const std::unordered_map<int, Node> &nodes,
        const std::vector<Connection> &connections);
};

#endif //UWU_LEARNER_TOPOLOGICALSORT_H
