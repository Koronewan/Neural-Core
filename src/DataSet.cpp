//
// Created by tar87 on 20/12/2024.
//
#include "DataSet.h"

#include <random>
#include <algorithm>
#include <sstream>

DataSet::DataSet(const std::vector<std::vector<double>> &features, const std::vector<std::vector<double>> &labels)
{
    if(features.size() != labels.size())
        throw std::invalid_argument("Features and Labels vectors don't have same size");

    this->items_ = static_cast<int>(features.size());
    this->features_ = features;
    this->labels_ = labels;
}

DataSet::DataSet(const DataSet &other)
{
    this->items_ = other.items_;
    this->features_ = other.features_;
    this->labels_ = other.labels_;
}

void DataSet::shuffle()
{
    std::vector<int> indexes(this->items_);

    for(int i = 0; i < this->items_; i++)
        indexes[i] = i;

    std::shuffle(indexes.begin(), indexes.end(), std::mt19937(std::random_device()()));

    std::vector<std::vector<double>> shuffledFeatures(this->items_);
    std::vector<std::vector<double>> shuffledLabels(this->items_);

    for(int i = 0; i < this->items_; i++)
    {
        shuffledFeatures[i] = this->features_[indexes[i]];
        shuffledLabels[i] = this->labels_[indexes[i]];
    }

    this->features_ = shuffledFeatures;
    this->labels_ = shuffledLabels;
}

void DataSet::normalize(double val)
{
    for (auto& feature : this->features_)
    {
        for (auto& value : feature)
        {
            value /= val;
        }
    }
}

std::pair<DataSet, DataSet> DataSet::split(double ratio) const
{
    const int leftSize = this->items_ - static_cast<int>(static_cast<double>(this->items_) * ratio);

    std::vector<std::vector<double>> leftFeatures(leftSize);
    std::vector<std::vector<double>> leftLabels(leftSize);

    for(int i = 0; i < leftSize; i++)
    {
        leftFeatures[i] = this->features_[i];
        leftLabels[i] = this->labels_[i];
    }

    int rightSize = this->items_ - leftSize;

    if (rightSize <= 0)
    {
        DataSet leftSet(leftFeatures, leftLabels);
        DataSet rightSet = DataSet();
        return std::make_pair(std::move(leftSet), std::move(rightSet));
    }

    std::vector<std::vector<double>> rightFeatures(rightSize);
    std::vector<std::vector<double>> rightLabels(rightSize);

    for(int i = leftSize; i < this->items_; i++)
    {
        rightFeatures[i - leftSize] = this->features_[i];
        rightLabels[i - leftSize] = this->labels_[i];
    }

    DataSet leftSet(leftFeatures, leftLabels);
    DataSet rightSet(rightFeatures, rightLabels);
    return std::make_pair(std::move(leftSet),std::move(rightSet));
}

Matrix DataSet::getFeatures() const
{
    return Matrix(this->features_);
}

Matrix DataSet::getLabels() const
{
    return Matrix(this->labels_);
}

int DataSet::getItems() const
{
    return this->items_;
}
