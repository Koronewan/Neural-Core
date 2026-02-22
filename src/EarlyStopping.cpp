//
// Created by korone on 12/21/24.
//

#include "EarlyStopping.h"

#include <iostream>
#include <limits>
#include <ostream>
#include <stdexcept>

EarlyStopping::EarlyStopping()
    : patience_(__INT_MAX__), delta_(0), stopped_(false), epochsWithoutImprovement_(0),
      bestMetric_(std::numeric_limits<double>::infinity()), mode_("min") {}

EarlyStopping::EarlyStopping(int patience, double delta, const std::string& mode)
    : patience_(patience), delta_(delta), stopped_(false),
      epochsWithoutImprovement_(0), mode_(mode) {
    if (mode_ == "min") {
        bestMetric_ = std::numeric_limits<double>::infinity();
    } else if (mode_ == "max") {
        bestMetric_ = -std::numeric_limits<double>::infinity();
    }
}

void EarlyStopping::reset() {
    epochsWithoutImprovement_ = 0;
    stopped_ = false;
    if (mode_ == "min") {
        bestMetric_ = std::numeric_limits<double>::infinity();
    } else {
        bestMetric_ = -std::numeric_limits<double>::infinity();
    }
}

void EarlyStopping::evaluate(double currentMetric) {
    if ((mode_ == "min" && currentMetric < bestMetric_ - delta_) ||
        (mode_ == "max" && currentMetric > bestMetric_ + delta_)) {
        bestMetric_ = currentMetric;
        epochsWithoutImprovement_ = 0;
        } else {
            epochsWithoutImprovement_++;
        }

    if (epochsWithoutImprovement_ > patience_) {
        stopped_ = true;
    }
}

bool EarlyStopping::shouldStop() const {
    return stopped_;
}

