//
// Created by aapr6 on 1/19/25.
// Created by korone on 1/19/25.
//

#include <gtest/gtest.h>
#include <fstream>
#include "../src/MathUtils/Matrix.h" // Incluye aquí la clase Matrix
#include "../src/MathUtils/Matrix.h"  // Include your Matrix class header

// Test case for operator+= with a scalar value
TEST(MatrixTest, ScalarAddition) {
    Matrix mat(2, 2);  // Create a 2x2 matrix
    mat += 5.0;  // Add 5.0 to each element

    // Check if all elements are now 5.0
    EXPECT_DOUBLE_EQ(mat.data_[0][0], 5.0);
    EXPECT_DOUBLE_EQ(mat.data_[0][1], 5.0);
    EXPECT_DOUBLE_EQ(mat.data_[1][0], 5.0);
    EXPECT_DOUBLE_EQ(mat.data_[1][1], 5.0);
}

// Test case for operator+= with another matrix
TEST(MatrixTest, MatrixAddition) {
    Matrix mat1(2, 2);
    mat1.data_ = {{1.0, 2.0}, {3.0, 4.0}};  // Initialize mat1 with values

    Matrix mat2(2, 2);
    mat2.data_ = {{5.0, 6.0}, {7.0, 8.0}};  // Initialize mat2 with values

    mat1 += mat2;  // Add mat2 to mat1

    // Check if the resulting mat1 contains the correct sum
    EXPECT_DOUBLE_EQ(mat1.data_[0][0], 6.0);
    EXPECT_DOUBLE_EQ(mat1.data_[0][1], 8.0);
    EXPECT_DOUBLE_EQ(mat1.data_[1][0], 10.0);
    EXPECT_DOUBLE_EQ(mat1.data_[1][1], 12.0);
}

// Test case for operator+= with matrices of mismatched dimensions
TEST(MatrixTest, MatrixAdditionDimensionMismatch) {
    Matrix mat1(2, 2);
    mat1.data_ = {{1.0, 2.0}, {3.0, 4.0}};  // Initialize mat1 with values

    Matrix mat2(3, 3);  // Mismatched dimensions
    mat2.data_ = {{5.0, 6.0, 7.0}, {8.0, 9.0, 10.0}, {11.0, 12.0, 13.0}};

    // This should throw an exception
    EXPECT_THROW(mat1 += mat2, std::invalid_argument);
}

// Test case for operator/= with a scalar value
TEST(MatrixTest, ScalarDivision) {
    Matrix mat(2, 2);  // Create a 2x2 matrix
    mat.data_ = {{4.0, 8.0}, {12.0, 16.0}};  // Initialize matrix with values
    mat /= 2;  // Divide each element by 2

    // Check if each element is divided by 2
    EXPECT_DOUBLE_EQ(mat.data_[0][0], 2.0);
    EXPECT_DOUBLE_EQ(mat.data_[0][1], 4.0);
    EXPECT_DOUBLE_EQ(mat.data_[1][0], 6.0);
    EXPECT_DOUBLE_EQ(mat.data_[1][1], 8.0);
}

// Test case for operator-= with another matrix
TEST(MatrixTest, MatrixSubtraction) {
    Matrix mat1(2, 2);
    mat1.data_ = {{5.0, 7.0}, {9.0, 11.0}};  // Initialize mat1 with values

    Matrix mat2(2, 2);
    mat2.data_ = {{1.0, 2.0}, {3.0, 4.0}};  // Initialize mat2 with values

    mat1 -= mat2;  // Subtract mat2 from mat1

    // Check if the resulting mat1 contains the correct difference
    EXPECT_DOUBLE_EQ(mat1.data_[0][0], 4.0);
    EXPECT_DOUBLE_EQ(mat1.data_[0][1], 5.0);
    EXPECT_DOUBLE_EQ(mat1.data_[1][0], 6.0);
    EXPECT_DOUBLE_EQ(mat1.data_[1][1], 7.0);
}

// Test case for operator-= with matrices of mismatched dimensions
TEST(MatrixTest, MatrixSubtractionDimensionMismatch) {
    Matrix mat1(2, 2);
    mat1.data_ = {{5.0, 7.0}, {9.0, 11.0}};  // Initialize mat1 with values

    Matrix mat2(3, 3);  // Mismatched dimensions
    mat2.data_ = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

    // This should throw an exception
    EXPECT_THROW(mat1 -= mat2, std::invalid_argument);
}

TEST(MatrixTest, ScalarMultiplication) {
    Matrix mat(2, 2);
    mat.data_ = {{1.0, 2.0}, {3.0, 4.0}};  // Initialize mat with values

    Matrix result = mat * 2.0;  // Multiply each element by 2.0

    // Check if each element is multiplied by 2.0
    EXPECT_DOUBLE_EQ(result.data_[0][0], 2.0);
    EXPECT_DOUBLE_EQ(result.data_[0][1], 4.0);
    EXPECT_DOUBLE_EQ(result.data_[1][0], 6.0);
    EXPECT_DOUBLE_EQ(result.data_[1][1], 8.0);
}

TEST(MatrixTest, MatrixSubtractionOperator) {
    Matrix mat1(2, 2);
    mat1.data_ = {{5.0, 7.0}, {9.0, 11.0}};  // Initialize mat1 with values

    Matrix mat2(2, 2);
    mat2.data_ = {{1.0, 2.0}, {3.0, 4.0}};  // Initialize mat2 with values

    Matrix result = mat1 - mat2;  // Subtract mat2 from mat1

    // Check if the resulting matrix contains the correct difference
    EXPECT_DOUBLE_EQ(result.data_[0][0], 4.0);
    EXPECT_DOUBLE_EQ(result.data_[0][1], 5.0);
    EXPECT_DOUBLE_EQ(result.data_[1][0], 6.0);
    EXPECT_DOUBLE_EQ(result.data_[1][1], 7.0);
}

TEST(MatrixTest, ScalarPowerOperator) {
    Matrix mat(2, 2);
    mat.data_ = {{2.0, 3.0}, {4.0, 5.0}};  // Initialize mat with values

    Matrix result = mat ^ 2.0;  // Raise each element to the power of 2.0

    // Check if each element is raised to the power of 2.0
    EXPECT_DOUBLE_EQ(result.data_[0][0], 4.0);  // 2^2 = 4
    EXPECT_DOUBLE_EQ(result.data_[0][1], 9.0);  // 3^2 = 9
    EXPECT_DOUBLE_EQ(result.data_[1][0], 16.0); // 4^2 = 16
    EXPECT_DOUBLE_EQ(result.data_[1][1], 25.0); // 5^2 = 25
}

TEST(MatrixTest, MatrixDivisionOperator) {
    Matrix mat1(2, 2);
    mat1.data_ = {{6.0, 8.0}, {10.0, 12.0}};  // Initialize mat1 with values

    Matrix mat2(2, 2);
    mat2.data_ = {{2.0, 4.0}, {5.0, 6.0}};  // Initialize mat2 with values

    Matrix result = mat1 / mat2;  // Divide each element of mat1 by the corresponding element of mat2

    // Check if each element of mat1 is divided by the corresponding element of mat2
    EXPECT_DOUBLE_EQ(result.data_[0][0], 3.0);  // 6/2 = 3
    EXPECT_DOUBLE_EQ(result.data_[0][1], 2.0);  // 8/4 = 2
    EXPECT_DOUBLE_EQ(result.data_[1][0], 2.0);  // 10/5 = 2
    EXPECT_DOUBLE_EQ(result.data_[1][1], 2.0);  // 12/6 = 2
}

TEST(MatrixTest, SaveAndLoadBinary) {
    // Crear una matriz de prueba
    Matrix original({
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    });

    // Archivo temporal para pruebas
    const std::string testFile = "test_matrix.bin";

    // Guardar la matriz en el archivo
    {
        std::ofstream outFile(testFile, std::ios::binary);
        ASSERT_TRUE(outFile.is_open()) << "No se pudo abrir el archivo para escritura.";
        original.saveToBinary(outFile);
    }

    // Cargar la matriz desde el archivo
    Matrix loaded;
    {
        std::ifstream inFile(testFile, std::ios::binary);
        ASSERT_TRUE(inFile.is_open()) << "No se pudo abrir el archivo para lectura.";
        loaded.loadFromBinary(inFile);
    }

    // Validar que las dimensiones de las matrices coinciden
    ASSERT_EQ(original.rows(), loaded.rows()) << "El número de filas no coincide.";
    ASSERT_EQ(original.columns(), loaded.columns()) << "El número de columnas no coincide.";

    // Validar que los datos de las matrices coinciden
    for (int i = 0; i < original.rows(); ++i) {
        for (int j = 0; j < original.columns(); ++j) {
            EXPECT_DOUBLE_EQ(original(i, j), loaded(i, j)) << "El valor en (" << i << ", " << j << ") no coincide.";
        }
    }

    // Limpiar el archivo de prueba
    std::remove(testFile.c_str());
}