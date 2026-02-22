//
// Created by korone on 12/21/24.
//

#include "Dropout.h"

Dropout::Dropout(double dropoutRate) : dropoutRatio_(dropoutRate) {}

uwu::Vector Dropout::forward(const uwu::Vector& input) {
    dropoutMask_ = uwu::Vector(input.size(), 1.0);

    std::mt19937 generator(std::random_device{}());
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    uwu::Vector output(input.size(), 0.0);

    for (std::size_t i = 0; i < input.size(); ++i) {
        if (distribution(generator) < dropoutRatio_) {
            dropoutMask_[i] = 0.0;
            output[i] = 0.0;
        } else {
            dropoutMask_[i] = 1.0;
            output[i] = input[i];
        }
    }

    return output;
}

void Dropout::backward(uwu::Vector &error, const uwu::Vector &previousActivation) {
    for (std::size_t i = 0; i < error.size(); ++i) {
        error[i] *= dropoutMask_[i]; // Escalar gradiente con la máscara
    }
}

void Dropout::update(const std::string &event) {

}

void Dropout::saveToBinary(std::ofstream &outFile) const {
    if (!outFile) {
        throw std::runtime_error("El archivo de salida no está abierto.");
    }

    // Guardar el tipo de capa
    std::string layerType = getType(); // "Dropout"
    size_t typeLength = layerType.size();
    outFile.write(reinterpret_cast<const char*>(&typeLength), sizeof(typeLength));
    outFile.write(layerType.data(), typeLength);

    // Guardar el dropoutRatio_
    outFile.write(reinterpret_cast<const char*>(&dropoutRatio_), sizeof(dropoutRatio_));

    dropoutMask_.saveToBinary(outFile);
}

void Dropout::loadFromBinary(std::ifstream &inFile) {
    if (!inFile) {
        throw std::runtime_error("El archivo de entrada no está abierto.");
    }

    // Leer el dropoutRatio_
    inFile.read(reinterpret_cast<char*>(&dropoutRatio_), sizeof(dropoutRatio_));
    if (!inFile) throw std::runtime_error("Error al leer el dropoutRatio_.");

    // Cargar el dropoutMask_ usando sus propios métodos
    dropoutMask_.loadFromBinary(inFile);
}

std::string Dropout::getInfo() const {
    std::ostringstream oss;
    oss << "Layer Type: Dropout\n";
    oss << "Dropout Ratio: " << dropoutRatio_ << "\n";
    return oss.str();
}
