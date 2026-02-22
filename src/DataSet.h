//
// Created by tar87 on 20/12/2024.
//

#ifndef UWU_LEARNER_DATASET_H
#define UWU_LEARNER_DATASET_H

#include "MathUtils/Matrix.h"
#include <vector>

class DataSet
{
    std::vector<std::vector<double>> features_;
    std::vector<std::vector<double>> labels_;
    int items_{};

public:
    DataSet(std::vector<std::vector<double>> const &features, std::vector<std::vector<double>> const &labels);
    DataSet(DataSet const &other);
    DataSet() = default;
    ~DataSet() = default;

    void shuffle();
    void normalize(double val);
    [[nodiscard]] std::pair<DataSet, DataSet> split(double ratio) const;

    [[nodiscard]] Matrix getFeatures() const;
    [[nodiscard]] Matrix getLabels() const;
    [[nodiscard]] int getItems() const;
};

#endif //UWU_LEARNER_DATASET_H
