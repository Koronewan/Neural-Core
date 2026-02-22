//
// Created by korone on 12/21/24.
//

#ifndef UWU_LEARNER_EARLYSTOPPING_H
#define UWU_LEARNER_EARLYSTOPPING_H

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

#endif //UWU_LEARNER_EARLYSTOPPING_H
