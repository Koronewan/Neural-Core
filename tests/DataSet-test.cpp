//
// Created by tar87 on 20/12/2024.
//

#include <gtest/gtest.h>
#include "../src/DataSet.h"

namespace {
    constexpr double HALF_SPLIT_RATIO = 0.5;
    constexpr double TRAIN_HEAVY_SPLIT_RATIO = 0.85;
    constexpr int SMALL_DATASET_SIZE = 2;
    constexpr int LARGE_DATASET_SIZE = 10;
    constexpr int EXPECTED_TRAIN_SIZE_85_PCT = 8;
    constexpr int EXPECTED_TEST_SIZE_85_PCT = 2;
}

TEST(DataSetTest, ConstructorTest)
{
    std::vector<std::vector<double>> features = {{1.0, 2.0}, {3.0, 4.0}};
    std::vector<std::vector<double>> labels = {{1.0}, {0.0}};

    DataSet dataset(features, labels);

    EXPECT_EQ(dataset.getItems(), SMALL_DATASET_SIZE);
}

TEST(DataSetTest, ShuffleTest)
{
    std::vector<std::vector<double>> features = {{1.0, 2.0}, {3.0, 4.0}};
    std::vector<std::vector<double>> labels = {{1.0}, {0.0}};
    DataSet dataset(features, labels);

    dataset.shuffle();
    EXPECT_EQ(dataset.getItems(), SMALL_DATASET_SIZE);
}

TEST(DataSetTest, SplitTest)
{
    std::vector<std::vector<double>> features = {{1.0, 2.0}, {3.0, 4.0}};
    std::vector<std::vector<double>> labels = {{1.0}, {0.0}};
    DataSet dataset(features, labels);

    auto [trainSet, testSet] = dataset.split(HALF_SPLIT_RATIO);

    constexpr int expectedSplitSize = SMALL_DATASET_SIZE / 2;
    EXPECT_EQ(trainSet.getItems(), expectedSplitSize);
    EXPECT_EQ(testSet.getItems(), expectedSplitSize);
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

    auto [trainSet, testSet] = dataset.split(TRAIN_HEAVY_SPLIT_RATIO);

    EXPECT_EQ(trainSet.getItems(), EXPECTED_TRAIN_SIZE_85_PCT);
    EXPECT_EQ(testSet.getItems(), EXPECTED_TEST_SIZE_85_PCT);
}
