//
// Created by aapr6 on 1/19/25.
// Created by korone on 1/19/25.
//

#include <gtest/gtest.h>
#include <fstream>
#include "../src/MathUtils/Matrix.h"

namespace {
    constexpr int MATRIX_2X2 = 2;
    constexpr int MATRIX_3X3 = 3;
    constexpr double SCALAR_ADDEND = 5.0;
    constexpr double SCALAR_DIVISOR = 2;
    constexpr double SCALAR_MULTIPLIER = 2.0;
    constexpr double SCALAR_EXPONENT = 2.0;
    const std::string BINARY_TEST_FILE = "test_matrix.bin";
}

TEST(MatrixTest, ScalarAddition) {
    Matrix mat(MATRIX_2X2, MATRIX_2X2);
    mat += SCALAR_ADDEND;

    // All elements start at 0, so result should be the scalar value
    EXPECT_DOUBLE_EQ(mat.data_[0][0], SCALAR_ADDEND);
    EXPECT_DOUBLE_EQ(mat.data_[0][1], SCALAR_ADDEND);
    EXPECT_DOUBLE_EQ(mat.data_[1][0], SCALAR_ADDEND);
    EXPECT_DOUBLE_EQ(mat.data_[1][1], SCALAR_ADDEND);
}

TEST(MatrixTest, MatrixAddition) {
    Matrix mat1(MATRIX_2X2, MATRIX_2X2);
    mat1.data_ = {{1.0, 2.0}, {3.0, 4.0}};

    Matrix mat2(MATRIX_2X2, MATRIX_2X2);
    mat2.data_ = {{5.0, 6.0}, {7.0, 8.0}};

    mat1 += mat2;

    EXPECT_DOUBLE_EQ(mat1.data_[0][0], 6.0);   // 1+5
    EXPECT_DOUBLE_EQ(mat1.data_[0][1], 8.0);   // 2+6
    EXPECT_DOUBLE_EQ(mat1.data_[1][0], 10.0);  // 3+7
    EXPECT_DOUBLE_EQ(mat1.data_[1][1], 12.0);  // 4+8
}

TEST(MatrixTest, MatrixAdditionDimensionMismatch) {
    Matrix mat1(MATRIX_2X2, MATRIX_2X2);
    mat1.data_ = {{1.0, 2.0}, {3.0, 4.0}};

    Matrix mat2(MATRIX_3X3, MATRIX_3X3);
    mat2.data_ = {{5.0, 6.0, 7.0}, {8.0, 9.0, 10.0}, {11.0, 12.0, 13.0}};

    EXPECT_THROW(mat1 += mat2, std::invalid_argument);
}

TEST(MatrixTest, ScalarDivision) {
    Matrix mat(MATRIX_2X2, MATRIX_2X2);
    mat.data_ = {{4.0, 8.0}, {12.0, 16.0}};
    mat /= SCALAR_DIVISOR;

    EXPECT_DOUBLE_EQ(mat.data_[0][0], 2.0);   // 4/2
    EXPECT_DOUBLE_EQ(mat.data_[0][1], 4.0);   // 8/2
    EXPECT_DOUBLE_EQ(mat.data_[1][0], 6.0);   // 12/2
    EXPECT_DOUBLE_EQ(mat.data_[1][1], 8.0);   // 16/2
}

TEST(MatrixTest, MatrixSubtraction) {
    Matrix mat1(MATRIX_2X2, MATRIX_2X2);
    mat1.data_ = {{5.0, 7.0}, {9.0, 11.0}};

    Matrix mat2(MATRIX_2X2, MATRIX_2X2);
    mat2.data_ = {{1.0, 2.0}, {3.0, 4.0}};

    mat1 -= mat2;

    EXPECT_DOUBLE_EQ(mat1.data_[0][0], 4.0);  // 5-1
    EXPECT_DOUBLE_EQ(mat1.data_[0][1], 5.0);  // 7-2
    EXPECT_DOUBLE_EQ(mat1.data_[1][0], 6.0);  // 9-3
    EXPECT_DOUBLE_EQ(mat1.data_[1][1], 7.0);  // 11-4
}

TEST(MatrixTest, MatrixSubtractionDimensionMismatch) {
    Matrix mat1(MATRIX_2X2, MATRIX_2X2);
    mat1.data_ = {{5.0, 7.0}, {9.0, 11.0}};

    Matrix mat2(MATRIX_3X3, MATRIX_3X3);
    mat2.data_ = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

    EXPECT_THROW(mat1 -= mat2, std::invalid_argument);
}

TEST(MatrixTest, ScalarMultiplication) {
    Matrix mat(MATRIX_2X2, MATRIX_2X2);
    mat.data_ = {{1.0, 2.0}, {3.0, 4.0}};

    Matrix result = mat * SCALAR_MULTIPLIER;

    EXPECT_DOUBLE_EQ(result.data_[0][0], 2.0);  // 1*2
    EXPECT_DOUBLE_EQ(result.data_[0][1], 4.0);  // 2*2
    EXPECT_DOUBLE_EQ(result.data_[1][0], 6.0);  // 3*2
    EXPECT_DOUBLE_EQ(result.data_[1][1], 8.0);  // 4*2
}

TEST(MatrixTest, MatrixSubtractionOperator) {
    Matrix mat1(MATRIX_2X2, MATRIX_2X2);
    mat1.data_ = {{5.0, 7.0}, {9.0, 11.0}};

    Matrix mat2(MATRIX_2X2, MATRIX_2X2);
    mat2.data_ = {{1.0, 2.0}, {3.0, 4.0}};

    Matrix result = mat1 - mat2;

    EXPECT_DOUBLE_EQ(result.data_[0][0], 4.0);  // 5-1
    EXPECT_DOUBLE_EQ(result.data_[0][1], 5.0);  // 7-2
    EXPECT_DOUBLE_EQ(result.data_[1][0], 6.0);  // 9-3
    EXPECT_DOUBLE_EQ(result.data_[1][1], 7.0);  // 11-4
}

TEST(MatrixTest, ScalarPowerOperator) {
    Matrix mat(MATRIX_2X2, MATRIX_2X2);
    mat.data_ = {{2.0, 3.0}, {4.0, 5.0}};

    Matrix result = mat ^ SCALAR_EXPONENT;

    EXPECT_DOUBLE_EQ(result.data_[0][0], 4.0);   // 2^2
    EXPECT_DOUBLE_EQ(result.data_[0][1], 9.0);   // 3^2
    EXPECT_DOUBLE_EQ(result.data_[1][0], 16.0);  // 4^2
    EXPECT_DOUBLE_EQ(result.data_[1][1], 25.0);  // 5^2
}

TEST(MatrixTest, MatrixDivisionOperator) {
    Matrix mat1(MATRIX_2X2, MATRIX_2X2);
    mat1.data_ = {{6.0, 8.0}, {10.0, 12.0}};

    Matrix mat2(MATRIX_2X2, MATRIX_2X2);
    mat2.data_ = {{2.0, 4.0}, {5.0, 6.0}};

    Matrix result = mat1 / mat2;

    EXPECT_DOUBLE_EQ(result.data_[0][0], 3.0);  // 6/2
    EXPECT_DOUBLE_EQ(result.data_[0][1], 2.0);  // 8/4
    EXPECT_DOUBLE_EQ(result.data_[1][0], 2.0);  // 10/5
    EXPECT_DOUBLE_EQ(result.data_[1][1], 2.0);  // 12/6
}
