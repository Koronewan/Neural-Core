//
// Created by korone on 1/11/25.
//

#include "../src/MathUtils/Vector.h"
#include <gtest/gtest.h>
#include <fstream>

namespace {
    const std::string BINARY_TEST_FILE = "test_vector.bin";
}


// Test constructing uwu::Vector<int> and checking contents
TEST(VectorTest, ConstructorInt) {
    std::vector baseData = {1.0, 2.0, 3.0};
    uwu::Vector myVec(baseData);

    // Check that sizes match
    EXPECT_EQ(myVec.size(), baseData.size());

    // Check each element
    for (size_t i = 0; i < baseData.size(); ++i) {
        EXPECT_EQ(myVec[i], baseData[i])
            << "Element at index " << i << " should match constructor input.";
    }
}

// Test operator= (copy assignment)
TEST(VectorTest, CopyAssignment) {
    std::vector<double> baseData = {3.14, 2.718, 1.414};
    uwu::Vector original(baseData);

    uwu::Vector copied({0.0}); // dummy initial data
    copied = original;

    // Check that the size and contents match
    EXPECT_EQ(copied.size(), original.size());
    for (size_t i = 0; i < original.size(); ++i) {
        EXPECT_DOUBLE_EQ(copied[i], original[i])
            << "Element at index " << i << " should match after copy assignment.";
    }
}

TEST(VectorTest, VectorAdditionAssignmentOperator) {
    uwu::Vector vec1;
    vec1.data_ = {1.0, 2.0, 3.0};

    uwu::Vector vec2;
    vec2.data_ = {4.0, 5.0, 6.0};

    vec1 += vec2;

    EXPECT_DOUBLE_EQ(vec1.data_[0], 5.0);   // 1+4
    EXPECT_DOUBLE_EQ(vec1.data_[1], 7.0);   // 2+5
    EXPECT_DOUBLE_EQ(vec1.data_[2], 9.0);   // 3+6
}

TEST(VectorTest, VectorMultiplicationAssignmentOperator) {
    uwu::Vector vec1;
    vec1.data_ = {1.0, 2.0, 3.0};

    uwu::Vector vec2;
    vec2.data_ = {4.0, 5.0, 6.0};

    vec1 *= vec2;

    EXPECT_DOUBLE_EQ(vec1.data_[0], 4.0);   // 1*4
    EXPECT_DOUBLE_EQ(vec1.data_[1], 10.0);  // 2*5
    EXPECT_DOUBLE_EQ(vec1.data_[2], 18.0);  // 3*6
}

TEST(VectorTest, VectorSubtractionAssignmentOperator) {
    uwu::Vector vec1;
    vec1.data_ = {5.0, 6.0, 7.0};

    uwu::Vector vec2;
    vec2.data_ = {1.0, 2.0, 3.0};

    vec1 -= vec2;

    EXPECT_DOUBLE_EQ(vec1.data_[0], 4.0);  // 5-1
    EXPECT_DOUBLE_EQ(vec1.data_[1], 4.0);  // 6-2
    EXPECT_DOUBLE_EQ(vec1.data_[2], 4.0);  // 7-3
}

TEST(VectorTest, VectorSubtractionOperator) {
    uwu::Vector vec1;
    vec1.data_ = {5.0, 6.0, 7.0};

    uwu::Vector vec2;
    vec2.data_ = {1.0, 2.0, 3.0};

    uwu::Vector result = vec1 - vec2;

    EXPECT_DOUBLE_EQ(result.data_[0], 4.0);  // 5-1
    EXPECT_DOUBLE_EQ(result.data_[1], 4.0);  // 6-2
    EXPECT_DOUBLE_EQ(result.data_[2], 4.0);  // 7-3
}

TEST(VectorTest, VectorAdditionOperator) {
    uwu::Vector vec1;
    vec1.data_ = {1.0, 2.0, 3.0};

    uwu::Vector vec2;
    vec2.data_ = {4.0, 5.0, 6.0};

    uwu::Vector result = vec1 + vec2;

    EXPECT_DOUBLE_EQ(result.data_[0], 5.0);  // 1+4
    EXPECT_DOUBLE_EQ(result.data_[1], 7.0);  // 2+5
    EXPECT_DOUBLE_EQ(result.data_[2], 9.0);  // 3+6
}

TEST(VectorTest, VectorAdditionWithScalarOperator) {
    uwu::Vector vec;
    vec.data_ = {1.0, 2.0, 3.0};

    constexpr double scalar = 5.0;
    uwu::Vector result = vec + scalar;

    EXPECT_DOUBLE_EQ(result.data_[0], 6.0);  // 1+5
    EXPECT_DOUBLE_EQ(result.data_[1], 7.0);  // 2+5
    EXPECT_DOUBLE_EQ(result.data_[2], 8.0);  // 3+5
}

TEST(VectorTest, VectorMultiplicationWithScalarOperator) {
    uwu::Vector vec;
    vec.data_ = {1.0, 2.0, 3.0};

    constexpr double scalar = 3.0;
    uwu::Vector result = vec * scalar;

    EXPECT_DOUBLE_EQ(result.data_[0], 3.0);  // 1*3
    EXPECT_DOUBLE_EQ(result.data_[1], 6.0);  // 2*3
    EXPECT_DOUBLE_EQ(result.data_[2], 9.0);  // 3*3
}

TEST(VectorTest, VectorDivisionByScalarOperator) {
    uwu::Vector vec;
    vec.data_ = {6.0, 8.0, 10.0};

    constexpr double scalar = 2.0;
    uwu::Vector result = vec / scalar;

    EXPECT_DOUBLE_EQ(result.data_[0], 3.0);  // 6/2
    EXPECT_DOUBLE_EQ(result.data_[1], 4.0);  // 8/2
    EXPECT_DOUBLE_EQ(result.data_[2], 5.0);  // 10/2
}

TEST(VectorTest, VectorDivisionByVectorOperator) {
    uwu::Vector vec1;
    vec1.data_ = {6.0, 8.0, 10.0};

    uwu::Vector vec2;
    vec2.data_ = {2.0, 4.0, 5.0};

    uwu::Vector result = vec1 / vec2;

    EXPECT_DOUBLE_EQ(result.data_[0], 3.0);  // 6/2
    EXPECT_DOUBLE_EQ(result.data_[1], 2.0);  // 8/4
    EXPECT_DOUBLE_EQ(result.data_[2], 2.0);  // 10/5
}

TEST(VectorTest, VectorExponentiationOperator) {
    uwu::Vector vec;
    vec.data_ = {2.0, 3.0, 4.0};

    constexpr int exponent = 2;
    uwu::Vector result = vec ^ exponent;

    EXPECT_DOUBLE_EQ(result.data_[0], 4.0);   // 2^2
    EXPECT_DOUBLE_EQ(result.data_[1], 9.0);   // 3^2
    EXPECT_DOUBLE_EQ(result.data_[2], 16.0);  // 4^2
}

TEST(VectorTest, VectorNegationOperator) {
    uwu::Vector vec;
    vec.data_ = {1.0, -2.0, 3.0};

    uwu::Vector result = -vec;

    EXPECT_DOUBLE_EQ(result.data_[0], -1.0);  // -(1)
    EXPECT_DOUBLE_EQ(result.data_[1], 2.0);   // -(-2)
    EXPECT_DOUBLE_EQ(result.data_[2], -3.0);  // -(3)
}
