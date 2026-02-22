//
// Created by korone on 12/21/24.
//

#ifndef UWU_LEARNER_INTERFACELAYER_H
#define UWU_LEARNER_INTERFACELAYER_H
#include <vector>
#include <sstream>

#include "../Events/InterfaceEventListener.h"
#include "../MathUtils/Vector.h"

class InterfaceLayer: public InterfaceEventListener
{
public:
    virtual uwu::Vector forward(const uwu::Vector &input) = 0;
    virtual void backward(
        uwu::Vector &error, const uwu::Vector &previousActivation) = 0;
    [[nodiscard]] virtual std::string getType() const = 0;

    virtual void saveToBinary(std::ofstream& outFile) const = 0;
    virtual void loadFromBinary(std::ifstream& inFile) = 0;
    [[nodiscard]] virtual std::string getInfo() const = 0;
};

#endif //UWU_LEARNER_INTERFACELAYER_H
