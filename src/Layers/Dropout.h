//
// Created by korone on 12/21/24.
//

#ifndef UWU_LEARNER_DROPOUT_H
#define UWU_LEARNER_DROPOUT_H
#include "InterfaceLayer.h"
#include "MathUtils/Vector.h"
#include <random>

class Dropout final: public InterfaceLayer
{
    double dropoutRatio_;
    uwu::Vector dropoutMask_;
public:
    explicit Dropout() = default;
    explicit Dropout(double dropoutRatio);
    ~Dropout() override = default;

    uwu::Vector forward(const uwu::Vector &input) override;
    void backward(
        uwu::Vector &error, const uwu::Vector &previousActivation) override;

    void update(const std::string &event) override;
    [[nodiscard]] std::string getType() const override{return "Dropout";}
    void saveToBinary(std::ofstream &outFile) const override;
    void loadFromBinary(std::ifstream &inFile) override;
    [[nodiscard]] double getDropoutRatio() const {
        return dropoutRatio_;
    }
    [[nodiscard]] uwu::Vector getMask() const {
        return dropoutMask_;
    }
    [[nodiscard]] std::string getInfo() const override;
};

#endif //UWU_LEARNER_DROPOUT_H
