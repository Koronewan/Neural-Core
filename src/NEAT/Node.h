//
// Created by korone on 1/13/25.
//

#ifndef UWU_LEARNER_NODE_H
#define UWU_LEARNER_NODE_H

#include <string>
#include <sstream>


enum NodeType
{
    Indefined = -1,
    Sensor = 0,
    Hidden = 1,
    Output = 2,
};


struct Node
{
    NodeType type;
    double value = 0.0;
    double bias = 0.0;
    int activation = 2;
    int id;

    explicit Node(const NodeType type, int id): type(type) , id(id) {}
    Node() : type(Indefined), id(-1) {}
    ~Node() = default;
};



#endif //UWU_LEARNER_NODE_H
