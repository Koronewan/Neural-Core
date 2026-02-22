//
// Created by aapr6 on 1/19/25.
//
#include <gtest/gtest.h>
#include <fstream>
#include <stdexcept>
#include "../src/Layers/WeightedLayer.h" // Asegúrate de que la ruta al archivo de cabecera es correcta
#include "../src/Layers/Activations/ReLU.h"          // Para las funciones de activación
#include "../src/Layers/Regularization/LassoRegression.h"  // Para los regularizadores
#include "../src/Layers/Initializers/OneInitializer.h"

TEST(WeightedLayerTest, SaveLoadBinary) {
    // Crear un objeto de WeightedLayer
    int numIn = 3, numOut = 2;
    ReLU* activationFunc = new ReLU();
    InterfaceInitializer* init = new OneInitializer();
    auto* originalLayer = new WeightedLayer(numIn, numOut, activationFunc, init);

    // Guardar el objeto en un archivo binario
    std::ofstream outFile("testLayer.bin", std::ios::binary);
    ASSERT_TRUE(outFile.is_open());
    originalLayer->saveToBinary(outFile);
    outFile.close();

    // Crear un nuevo objeto de WeightedLayer para cargar los datos
    auto* loadedLayer = new WeightedLayer(numIn, numOut, activationFunc, init);

    // Cargar el objeto desde el archivo binario
    std::ifstream inFile("testLayer.bin", std::ios::binary);
    ASSERT_TRUE(inFile.is_open());
    loadedLayer->loadFromBinary(inFile);
    inFile.close();

    // Verificar que los datos cargados coinciden con los originales
    ASSERT_EQ(originalLayer->getWeights(), loadedLayer->getWeights());
    ASSERT_EQ(originalLayer->getBiases(), loadedLayer->getBiases());

    // Verificar que el tipo de función de activación y regularizador se cargaron correctamente
    ASSERT_EQ(originalLayer->getType(), loadedLayer->getType()); // Compara el tipo de capa
}