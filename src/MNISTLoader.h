//
// Created by korone on 1/16/25.
//

#ifndef MNISTLOADER_H
#define MNISTLOADER_H

#include <vector>
#include <string>

class MNISTLoader {
public:
    // Load the MNIST dataset
    bool load(const std::string& datasetPath);

    // Accessors for the training data
    std::vector<std::vector<double>> getTrainingImages() const { return trainingImages; }
    std::vector<std::vector<double>> getTrainingLabels() const { return trainingLabels; }

private:
    std::vector<std::vector<double>> trainingImages; // Flattened 28x28 images
    std::vector<std::vector<double>> trainingLabels; // One-hot encoded labels

    // Helper functions
    bool loadImages(const std::string& filePath, std::vector<std::vector<double>>& images);
    bool loadLabels(const std::string& filePath, std::vector<std::vector<double>>& labels);
};

#endif // MNISTLOADER_H
