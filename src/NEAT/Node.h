//
// Created by korone on 1/13/25.
//

#ifndef NEURAL_CORE_NODE_H
#define NEURAL_CORE_NODE_H

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



#endif //NEURAL_CORE_NODE_H
