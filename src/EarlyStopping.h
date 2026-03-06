//
// Created by korone on 12/21/24.
//

#ifndef NEURAL_CORE_EARLYSTOPPING_H
#define NEURAL_CORE_EARLYSTOPPING_H

#include <string>

class EarlyStopping {
    int patience_;
    double delta_;
    bool stopped_;
    int epochsWithoutImprovement_;
    double bestMetric_;
    std::string mode_; // 'min' para minimizar o 'max' para maximizar

public:
    EarlyStopping();
    EarlyStopping(int patience, double delta, const std::string& mode);
    void reset();
    void evaluate(double currentMetric);
    [[nodiscard]] bool shouldStop() const;
};

#endif //NEURAL_CORE_EARLYSTOPPING_H
