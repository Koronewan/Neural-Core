//
// Created by korone on 1/10/25.
//

#include "MeanSquarredError.h"

uwu::Vector MeanSquarredError::gradient(const uwu::Vector &item, const uwu::Vector &expectedItem)
{
    return - (expectedItem - item) / static_cast<double>(item.size());
}