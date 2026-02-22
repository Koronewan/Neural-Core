//
// Created by aapr6 on 1/12/25.
//

#include "CrossEntropy.h"

uwu::Vector CrossEntropy::gradient(const uwu::Vector &item, const uwu::Vector &expectedItem)
{
    return item - expectedItem;
}