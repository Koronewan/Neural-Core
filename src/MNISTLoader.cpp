//
// Created by korone on 1/16/25.
//

#include "MNISTLoader.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstdint>

// Helper function to reverse byte order for multi-byte integers
uint32_t reverseBytes(uint32_t value) {
    return ((value & 0xFF000000) >> 24) |
           ((value & 0x00FF0000) >> 8) |
           ((value & 0x0000FF00) << 8) |
           ((value & 0x000000FF) << 24);
}

bool MNISTLoader::load(const std::string& datasetPath) {
    const std::string trainingImagesPath = datasetPath + "/train-images-idx3-ubyte";
    const std::string trainingLabelsPath = datasetPath + "/train-labels-idx1-ubyte";

    return loadImages(trainingImagesPath, trainingImages) && loadLabels(trainingLabelsPath, trainingLabels);
}

bool MNISTLoader::loadImages(const std::string& filePath, std::vector<std::vector<double>>& images) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return false;
    }

    uint32_t magicNumber, numImages, rows, cols;
    file.read(reinterpret_cast<char*>(&magicNumber), 4);
    file.read(reinterpret_cast<char*>(&numImages), 4);
    file.read(reinterpret_cast<char*>(&rows), 4);
    file.read(reinterpret_cast<char*>(&cols), 4);

    magicNumber = reverseBytes(magicNumber);
    numImages = reverseBytes(numImages);
    rows = reverseBytes(rows);
    cols = reverseBytes(cols);

    if (magicNumber != 2051) { // Magic number for image files
        std::cerr << "Invalid magic number in image file: " << filePath << std::endl;
        return false;
    }

    images.resize(numImages, std::vector<double>(rows * cols));
    for (uint32_t i = 0; i < numImages; ++i) {
        std::vector<double>& image = images[i];
        for (uint32_t j = 0; j < rows * cols; ++j) {
            uint8_t pixel;
            file.read(reinterpret_cast<char*>(&pixel), 1);
            image[j] = static_cast<double>(pixel) / 255.0; // Normalize to [0, 1]
        }
    }

    return true;
}

bool MNISTLoader::loadLabels(const std::string& filePath, std::vector<std::vector<double>>& labels) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return false;
    }

    uint32_t magicNumber, numLabels;
    file.read(reinterpret_cast<char*>(&magicNumber), 4);
    file.read(reinterpret_cast<char*>(&numLabels), 4);

    magicNumber = reverseBytes(magicNumber);
    numLabels = reverseBytes(numLabels);

    if (magicNumber != 2049) { // Magic number for label files
        std::cerr << "Invalid magic number in label file: " << filePath << std::endl;
        return false;
    }

    labels.resize(numLabels, std::vector<double>(10, 0.0)); // One-hot encoding for 10 classes
    for (uint32_t i = 0; i < numLabels; ++i) {
        uint8_t label;
        file.read(reinterpret_cast<char*>(&label), 1);
        labels[i][label] = 1.0;
    }

    return true;
}
