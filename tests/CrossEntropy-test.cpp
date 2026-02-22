//
// Created by aapr6 on 1/12/25.
//

#include "gtest/gtest.h"
#include "../src/Loss/CrossEntropy.h"

TEST(CrossEntropyTest, GradientComputation)
{
    // Valores de entrada (predicción y valor esperado)
    std::vector<double> itemData = {0.1, 0.9, 0.8};
    uwu::Vector item = uwu::Vector(itemData);

    std::vector<double> expectedItemData = {0.0, 1.0, 0.0};
    uwu::Vector expectedItem = uwu::Vector(expectedItemData);

    // Resultado esperado del gradiente
    std::vector<double> expectedGradientData = {-0.0, -1.1111111, -0.0};  // Ajustar según la fórmula

    // Crear instancia de CrossEntropy
    CrossEntropy lossFunction;

    // Calcular el gradiente
    uwu::Vector computedGradient = lossFunction.gradient(item, expectedItem);

    // Verificar que los valores del gradiente sean correctos con una tolerancia
    double tolerance = 1e-5;
    for (size_t i = 0; i < computedGradient.size(); ++i) {
        EXPECT_NEAR(computedGradient[i], expectedGradientData[i], tolerance);
    }
}

// Test para el gradiente de CrossEntropy con predicciones y valores esperados completamente diferentes
TEST(CrossEntropyTest, GradientDifferentPredictions)
{
    // Valores de entrada (predicción y valor esperado completamente diferentes)
    std::vector<double> itemData = {0.05, 0.7, 0.2};
    uwu::Vector item = uwu::Vector(itemData);

    std::vector<double> expectedItemData = {1.0, 0.0, 1.0};
    uwu::Vector expectedItem = uwu::Vector(expectedItemData);

    // Resultado esperado del gradiente
    std::vector<double> expectedGradientData = {-19.999996, 0.0, -4.99999999};  // Ajustar según la fórmula

    // Crear instancia de CrossEntropy
    CrossEntropy lossFunction;

    // Calcular el gradiente
    uwu::Vector computedGradient = lossFunction.gradient(item, expectedItem);

    // Verificar que los valores del gradiente sean correctos con una tolerancia
    double tolerance = 1e-5;
    for (int i = 0; i < computedGradient.size(); ++i) {
        EXPECT_NEAR(computedGradient[i], expectedGradientData[i], tolerance);
    }
}

// Test para el gradiente de CrossEntropy con valores muy pequeños
TEST(CrossEntropyTest, GradientWithSmallPredictions)
{
    // Valores de entrada (predicciones pequeñas)
    std::vector<double> itemData = {1e-10, 1e-10, 1e-10};
    uwu::Vector item = uwu::Vector(itemData);

    std::vector<double> expectedItemData = {1.0, 0.0, 1.0};
    uwu::Vector expectedItem = uwu::Vector(expectedItemData);

    // Resultado esperado del gradiente
    std::vector<double> expectedGradientData = {-99009900.990099013, 0, -99009900.990099013};  // Ajustar según la fórmula

    // Crear instancia de CrossEntropy
    CrossEntropy lossFunction;

    // Calcular el gradiente
    uwu::Vector computedGradient = lossFunction.gradient(item, expectedItem);

    // Verificar que los valores del gradiente sean correctos con una tolerancia
    double tolerance = 1e-5;
    for (int i = 0; i < computedGradient.size(); ++i) {
        EXPECT_NEAR(computedGradient[i], expectedGradientData[i], tolerance);
    }
}

// Test para el gradiente de CrossEntropy con predicciones cercanas a 1
TEST(CrossEntropyTest, GradientWithPredictionsCloseToOne)
{
    // Valores de entrada (predicciones cercanas a 1)
    std::vector<double> itemData = {0.999, 0.95, 0.999};
    uwu::Vector item = uwu::Vector(itemData);

    std::vector<double> expectedItemData = {1.0, 1.0, 0.0};
    uwu::Vector expectedItem = uwu::Vector(expectedItemData);

    // Resultado esperado del gradiente
    std::vector<double> expectedGradientData = {-1.001000991, -1.052631568, 0};  // Ajustar según la fórmula

    // Crear instancia de CrossEntropy
    CrossEntropy lossFunction;

    // Calcular el gradiente
    uwu::Vector computedGradient = lossFunction.gradient(item, expectedItem);

    // Verificar que los valores del gradiente sean correctos con una tolerancia
    double tolerance = 1e-5;
    for (int i = 0; i < computedGradient.size(); ++i) {
        EXPECT_NEAR(computedGradient[i], expectedGradientData[i], tolerance);
    }
}

// Test para el gradiente de CrossEntropy con predicciones de 0
TEST(CrossEntropyTest, GradientWithPredictionsZero)
{
    // Valores de entrada (predicciones de 0)
    std::vector<double> itemData = {0.0, 0.0, 0.0};
    uwu::Vector item = uwu::Vector(itemData);

    std::vector<double> expectedItemData = {1.0, 0.0, 1.0};
    uwu::Vector expectedItem = uwu::Vector(expectedItemData);

    // Resultado esperado del gradiente
    std::vector<double> expectedGradientData = {-1e8, 0.0, -1e8};  // Ajustar según la fórmula

    // Crear instancia de CrossEntropy
    CrossEntropy lossFunction;

    // Calcular el gradiente
    uwu::Vector computedGradient = lossFunction.gradient(item, expectedItem);

    // Verificar que los valores del gradiente sean correctos con una tolerancia
    double tolerance = 1e-5;
    for (int i = 0; i < computedGradient.size(); ++i) {
        EXPECT_NEAR(computedGradient[i], expectedGradientData[i], tolerance);
    }
}