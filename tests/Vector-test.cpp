//
// Created by korone on 1/11/25.
//

#include "../src/MathUtils/Vector.h"
#include <gtest/gtest.h>


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
    vec1.data_ = {1.0, 2.0, 3.0};  // Initialize vec1 with values

    uwu::Vector vec2;
    vec2.data_ = {4.0, 5.0, 6.0};  // Initialize vec2 with values

    vec1 += vec2;  // Add vec2 to vec1

    // Check if the elements are added correctly
    EXPECT_DOUBLE_EQ(vec1.data_[0], 5.0);  // 1.0 + 4.0 = 5.0
    EXPECT_DOUBLE_EQ(vec1.data_[1], 7.0);  // 2.0 + 5.0 = 7.0
    EXPECT_DOUBLE_EQ(vec1.data_[2], 9.0);  // 3.0 + 6.0 = 9.0
}

TEST(VectorTest, VectorMultiplicationAssignmentOperator) {
    uwu::Vector vec1;
    vec1.data_ = {1.0, 2.0, 3.0};  // Initialize vec1 with values

    uwu::Vector vec2;
    vec2.data_ = {4.0, 5.0, 6.0};  // Initialize vec2 with values

    vec1 *= vec2;  // Multiply each element of vec1 by the corresponding element of vec2

    // Check if the elements are multiplied correctly
    EXPECT_DOUBLE_EQ(vec1.data_[0], 4.0);  // 1.0 * 4.0 = 4.0
    EXPECT_DOUBLE_EQ(vec1.data_[1], 10.0); // 2.0 * 5.0 = 10.0
    EXPECT_DOUBLE_EQ(vec1.data_[2], 18.0); // 3.0 * 6.0 = 18.0
}

TEST(VectorTest, VectorSubtractionAssignmentOperator) {
    uwu::Vector vec1;
    vec1.data_ = {5.0, 6.0, 7.0};  // Initialize vec1 with values

    uwu::Vector vec2;
    vec2.data_ = {1.0, 2.0, 3.0};  // Initialize vec2 with values

    vec1 -= vec2;  // Subtract vec2 from vec1

    // Check if the elements are subtracted correctly
    EXPECT_DOUBLE_EQ(vec1.data_[0], 4.0);  // 5.0 - 1.0 = 4.0
    EXPECT_DOUBLE_EQ(vec1.data_[1], 4.0);  // 6.0 - 2.0 = 4.0
    EXPECT_DOUBLE_EQ(vec1.data_[2], 4.0);  // 7.0 - 3.0 = 4.0
}

TEST(VectorTest, VectorSubtractionOperator) {
    uwu::Vector vec1;
    vec1.data_ = {5.0, 6.0, 7.0};  // Initialize vec1 with values

    uwu::Vector vec2;
    vec2.data_ = {1.0, 2.0, 3.0};  // Initialize vec2 with values

    uwu::Vector result = vec1 - vec2;  // Subtract vec2 from vec1

    // Check if the elements are subtracted correctly
    EXPECT_DOUBLE_EQ(result.data_[0], 4.0);  // 5.0 - 1.0 = 4.0
    EXPECT_DOUBLE_EQ(result.data_[1], 4.0);  // 6.0 - 2.0 = 4.0
    EXPECT_DOUBLE_EQ(result.data_[2], 4.0);  // 7.0 - 3.0 = 4.0
}

TEST(VectorTest, VectorAdditionOperator) {
    uwu::Vector vec1;
    vec1.data_ = {1.0, 2.0, 3.0};  // Initialize vec1 with values

    uwu::Vector vec2;
    vec2.data_ = {4.0, 5.0, 6.0};  // Initialize vec2 with values

    uwu::Vector result = vec1 + vec2;  // Add vec1 and vec2

    // Check if the elements are added correctly
    EXPECT_DOUBLE_EQ(result.data_[0], 5.0);  // 1.0 + 4.0 = 5.0
    EXPECT_DOUBLE_EQ(result.data_[1], 7.0);  // 2.0 + 5.0 = 7.0
    EXPECT_DOUBLE_EQ(result.data_[2], 9.0);  // 3.0 + 6.0 = 9.0
}

TEST(VectorTest, VectorAdditionWithScalarOperator) {
    uwu::Vector vec;
    vec.data_ = {1.0, 2.0, 3.0};  // Initialize vec with values

    double scalar = 5.0;

    uwu::Vector result = vec + scalar;  // Add scalar to each element of vec

    // Check if each element is incremented by the scalar
    EXPECT_DOUBLE_EQ(result.data_[0], 6.0);  // 1.0 + 5.0 = 6.0
    EXPECT_DOUBLE_EQ(result.data_[1], 7.0);  // 2.0 + 5.0 = 7.0
    EXPECT_DOUBLE_EQ(result.data_[2], 8.0);  // 3.0 + 5.0 = 8.0
}

TEST(VectorTest, VectorMultiplicationWithScalarOperator) {
    uwu::Vector vec;
    vec.data_ = {1.0, 2.0, 3.0};  // Initialize vec with values

    double scalar = 3.0;

    uwu::Vector result = vec * scalar;  // Multiply each element of vec by scalar

    // Check if each element is multiplied by the scalar
    EXPECT_DOUBLE_EQ(result.data_[0], 3.0);  // 1.0 * 3.0 = 3.0
    EXPECT_DOUBLE_EQ(result.data_[1], 6.0);  // 2.0 * 3.0 = 6.0
    EXPECT_DOUBLE_EQ(result.data_[2], 9.0);  // 3.0 * 3.0 = 9.0
}

TEST(VectorTest, VectorDivisionByScalarOperator) {
    uwu::Vector vec;
    vec.data_ = {6.0, 8.0, 10.0};  // Initialize vec with values

    double scalar = 2.0;

    uwu::Vector result = vec / scalar;  // Divide each element of vec by scalar

    // Check if each element is divided by the scalar
    EXPECT_DOUBLE_EQ(result.data_[0], 3.0);  // 6.0 / 2.0 = 3.0
    EXPECT_DOUBLE_EQ(result.data_[1], 4.0);  // 8.0 / 2.0 = 4.0
    EXPECT_DOUBLE_EQ(result.data_[2], 5.0);  // 10.0 / 2.0 = 5.0
}

TEST(VectorTest, VectorDivisionByVectorOperator) {
    uwu::Vector vec1;
    vec1.data_ = {6.0, 8.0, 10.0};  // Initialize vec1 with values

    uwu::Vector vec2;
    vec2.data_ = {2.0, 4.0, 5.0};  // Initialize vec2 with values

    uwu::Vector result = vec1 / vec2;  // Divide each element of vec1 by corresponding element of vec2

    // Check if the elements are divided correctly
    EXPECT_DOUBLE_EQ(result.data_[0], 3.0);  // 6.0 / 2.0 = 3.0
    EXPECT_DOUBLE_EQ(result.data_[1], 2.0);  // 8.0 / 4.0 = 2.0
    EXPECT_DOUBLE_EQ(result.data_[2], 2.0);  // 10.0 / 5.0 = 2.0
}

TEST(VectorTest, VectorExponentiationOperator) {
    uwu::Vector vec;
    vec.data_ = {2.0, 3.0, 4.0};  // Initialize vec with values

    int exponent = 2;

    uwu::Vector result = vec ^ exponent;  // Raise each element of vec to the power of 2

    // Check if the elements are raised to the correct power
    EXPECT_DOUBLE_EQ(result.data_[0], 4.0);  // 2.0 ^ 2 = 4.0
    EXPECT_DOUBLE_EQ(result.data_[1], 9.0);  // 3.0 ^ 2 = 9.0
    EXPECT_DOUBLE_EQ(result.data_[2], 16.0); // 4.0 ^ 2 = 16.0
}

TEST(VectorTest, VectorNegationOperator) {
    uwu::Vector vec;
    vec.data_ = {1.0, -2.0, 3.0};  // Initialize vec with values

    uwu::Vector result = -vec;  // Negate each element of vec

    // Check if each element is negated correctly
    EXPECT_DOUBLE_EQ(result.data_[0], -1.0);  // -1.0
    EXPECT_DOUBLE_EQ(result.data_[1], 2.0);   // -(-2.0) = 2.0
    EXPECT_DOUBLE_EQ(result.data_[2], -3.0);  // -3.0
}

TEST(VectorTest, SaveAndLoadBinary) {
    // Vector original para guardar
    uwu::Vector original({1.0, 2.5, 3.75});

    // Archivo temporal para pruebas
    const std::string testFile = "test_vector.bin";

    // Guardar el vector en el archivo
    {
        std::ofstream outFile(testFile, std::ios::binary);
        ASSERT_TRUE(outFile.is_open()) << "No se pudo abrir el archivo para escritura.";
        original.saveToBinary(outFile);
    }

    // Cargar el vector desde el archivo
    uwu::Vector loaded;
    {
        std::ifstream inFile(testFile, std::ios::binary);
        ASSERT_TRUE(inFile.is_open()) << "No se pudo abrir el archivo para lectura.";
        loaded.loadFromBinary(inFile);
    }

    // Comparar el vector original con el cargado
    ASSERT_EQ(original.size(), loaded.size()) << "El tamaño del vector cargado no coincide.";
    for (std::size_t i = 0; i < original.size(); ++i) {
        EXPECT_DOUBLE_EQ(original[i], loaded[i]) << "Los valores en el índice " << i << " no coinciden.";
    }

    // Limpiar el archivo de prueba
    std::remove(testFile.c_str());
}