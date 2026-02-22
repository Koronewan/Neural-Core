#include <gtest/gtest.h>
#include "../src/Layers/Dropout.h"


TEST(DropoutTest, ForwardDropoutMask) {
    Dropout dropout(0.5); // 50% de dropout
    std::vector<double> input_data = {1.0, 2.0, 3.0, 4.0, 5.0};
    uwu::Vector input(input_data);
    uwu::Vector output = dropout.forward(input);

    ASSERT_EQ(input.size(), output.size());

    int activeNeurons = 0;
    for (std::size_t i = 0; i < output.size(); ++i) {
        if (output[i] != 0.0) {
            activeNeurons++;
            EXPECT_EQ(input[i], output[i]);
        }
    }

    // Aproximadamente la mitad de las neuronas deben estar activas
    EXPECT_NEAR(static_cast<double>(activeNeurons) / input.size(), 0.5, 0.2);
}

TEST(DropoutTest, BackwardRespectsMask) {
    Dropout dropout(0.5);

    std::vector<double> input_data = {1.0, 2.0, 3.0, 4.0, 5.0};
    uwu::Vector input(input_data);
    uwu::Vector forwardOutput = dropout.forward(input);

    std::vector<double> error_data = {0.1, 0.2, 0.3, 0.4, 0.5};
    uwu::Vector error(error_data);
    uwu::Vector originalError = error; // Guardar copia para comparación

    dropout.backward(error, input);

    ASSERT_EQ(error.size(), forwardOutput.size());

    for (std::size_t i = 0; i < error.size(); ++i) {
        if (forwardOutput[i] == 0.0) {
            EXPECT_EQ(error[i], 0.0); // Error debe apagarse donde forward apagó neuronas
        } else {
            EXPECT_EQ(error[i], originalError[i]); // Error debe permanecer igual donde forward no apagó
        }
    }
}

TEST(DropoutTest, NoDropoutAtZeroRatio) {
    Dropout dropout(0.0); // No se apagan neuronas

    std::vector<double> input_data = {1.0, 2.0, 3.0, 4.0, 5.0};
    uwu::Vector input(input_data);
    uwu::Vector output = dropout.forward(input);

    ASSERT_EQ(input.size(), output.size());

    for (std::size_t i = 0; i < input.size(); ++i) {
        EXPECT_EQ(output[i], input[i]); // Ninguna neurona debe apagarse
    }
}

TEST(DropoutTest, FullDropoutAtOneRatio) {
    Dropout dropout(1.0); // Todas las neuronas se apagan

    std::vector<double> input_data = {1.0, 2.0, 3.0, 4.0, 5.0};
    uwu::Vector input(input_data);
    uwu::Vector output = dropout.forward(input);

    ASSERT_EQ(input.size(), output.size());

    for (std::size_t i = 0; i < input.size(); ++i) {
        EXPECT_EQ(output[i], 0.0); // Todas las salidas deben ser 0
    }
}

TEST(DropoutTest, SaveLoadBinary) {
    // Crear un objeto de Dropout
    double dropoutRatio = 0.5;
    Dropout originalLayer(dropoutRatio);

    // Simular un forward para generar un dropoutMask no vacío
    uwu::Vector input({1.0, 2.0, 3.0});
    uwu::Vector output = originalLayer.forward(input);

    // Guardar el objeto en un archivo binario
    std::ofstream outFile("testDropout.bin", std::ios::binary);
    ASSERT_TRUE(outFile.is_open());
    originalLayer.saveToBinary(outFile);
    outFile.close();

    // Crear un nuevo objeto de Dropout para cargar los datos
    Dropout loadedLayer(0.0); // Inicializar con un ratio diferente para asegurarnos de que se sobrescribe

    // Cargar el objeto desde el archivo binario
    std::ifstream inFile("testDropout.bin", std::ios::binary);
    ASSERT_TRUE(inFile.is_open());
    loadedLayer.loadFromBinary(inFile);
    inFile.close();

    // Verificar que el dropoutRatio cargado coincide con el original
    ASSERT_DOUBLE_EQ(originalLayer.getDropoutRatio(), loadedLayer.getDropoutRatio());

    // Verificar que el dropoutMask cargado coincide con el original
    const auto& originalMask = originalLayer.getMask();
    const auto& loadedMask = loadedLayer.getMask();
    ASSERT_EQ(originalMask.size(), loadedMask.size());
    for (size_t i = 0; i < originalMask.size(); ++i) {
        ASSERT_DOUBLE_EQ(originalMask[i], loadedMask[i]);
    }
}