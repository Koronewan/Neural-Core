//
// Created by tar87 on 20/12/2024.
//

#include <gtest/gtest.h>
#include "../src/DataSet.h"

TEST(DataSetTest, ConstructorTest)
{
    std::vector<std::vector<double>> features = {{1.0, 2.0}, {3.0, 4.0}};
    std::vector<std::vector<double>> labels = {{1.0}, {0.0}};

    DataSet dataset(features, labels);

    EXPECT_EQ(dataset.getItems(), 2);
}

TEST(DataSetTest, ShuffleTest)
{
    std::vector<std::vector<double>> features = {{1.0, 2.0}, {3.0, 4.0}};
    std::vector<std::vector<double>> labels = {{1.0}, {0.0}};
    DataSet dataset(features, labels);

    dataset.shuffle();
    EXPECT_EQ(dataset.getItems(), 2);
}

TEST(DataSetTest, SplitTest)
{
    std::vector<std::vector<double>> features = {{1.0, 2.0}, {3.0, 4.0}};
    std::vector<std::vector<double>> labels = {{1.0}, {0.0}};
    DataSet dataset(features, labels);

    auto [trainSet, testSet] = dataset.split(0.5);

    EXPECT_EQ(trainSet.getItems(), 1);
    EXPECT_EQ(testSet.getItems(), 1);
}

TEST(DataSetTest, SplitTestBigger)
{
    std::vector<std::vector<double>> features = {
        {1.0, 2.0}, {3.0, 4.0}, {3.0, 4.0}, {3.0, 4.0}, {3.0, 4.0},
        {1.0, 2.0}, {3.0, 4.0}, {3.0, 4.0}, {3.0, 4.0}, {3.0, 4.0}
        };
    std::vector<std::vector<double>> labels = {
        {1.0}, {0.0}, {1.0}, {0.0}, {1.0},
        {1.0}, {0.0}, {1.0}, {0.0}, {1.0}
        };

    DataSet dataset(features, labels);

    auto [trainSet, testSet] = dataset.split(0.85);

    EXPECT_EQ(trainSet.getItems(), 8);
    EXPECT_EQ(testSet.getItems(), 2);
}
