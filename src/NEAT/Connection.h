//
// Created by korone on 1/13/25.
//

#ifndef NEURAL_CORE_CONNECTION_H
#define NEURAL_CORE_CONNECTION_H

#include <string>
#include <sstream>

struct  Connection
{
    int inputNode;
    int outputNode;
    double weight;
    bool enabled;
    int innovation;

    Connection(const int inputNode, const int outputNode, const double weight, const int innovation):
        inputNode(inputNode),outputNode(outputNode), weight(weight), enabled(true), innovation(innovation) {}

    ~Connection() = default;

};

#endif //NEURAL_CORE_CONNECTION_H
