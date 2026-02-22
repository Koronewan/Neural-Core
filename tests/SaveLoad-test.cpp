//
// Created by aapr6 on 1/20/25.
//

#include "NEAT/SelfEvolvingNeuralNetwork.h"
#include "NeuralNetwork.h"
#include "Layers/Activations/ReLU.h"
#include "Layers/Activations/Sigmoid.h"
#include "Layers/Initializers/GlorotInitializer.h"
#include "Loss/MeanSquarredError.h"
#include "Metrics/Accuracy.h"
#include "Optimizers/Adam/Adam.h"
#include "Optimizers/RMSProp/RMSProp.h"
#include "Optimizers/SGD/SGD.h"
#include "MNISTLoader.h"
#include "Layers/Initializers/HeInitializer.h"
#include "Layers/Initializers/OneInitializer.h"
#include "Layers/Regularization/LassoRidgeRegression.h"
#include "Loss/CrossEntropy.h"
#include <gtest/gtest.h>

#include "Layers/Dropout.h"

TEST(NeuralNetworkTest, SaveAndLoadLayers) {
    // Crear red inicial
    NeuralNetwork network;
    network.addLayer(new WeightedLayer(3, 2, new Sigmoid(), new OneInitializer(), new RidgeRegression()));
    network.addLayer(new WeightedLayer(2, 1, new Sigmoid(), new OneInitializer(), new RidgeRegression()));

    InterfaceOptimizer* adam = new SGD(3);
    InterfaceLossFunction* mae = new MeanSquarredError();
    InterfaceMetric* accuracy = new Accuracy();

    network.compile(adam, mae, accuracy);

    // Guardar información de las capas
    std::string originalInfo = network.getLayersInfo();

    // Guardar red en archivo binario
    network.saveBinary("test_model.bin");

    // Crear nueva red y cargar desde el archivo binario
    NeuralNetwork loadedNetwork;
    loadedNetwork.loadBinary("test_model.bin");

    // Verificar que la información de las capas sea igual
    std::string loadedInfo = loadedNetwork.getLayersInfo();
    EXPECT_EQ(originalInfo, loadedInfo);

    // Liberar memoria
    delete adam;
    delete mae;
    delete accuracy;
}

TEST(NeuralNetworkTest, SaveAndLoadLayersHe) {
    // Crear red inicial
    NeuralNetwork network;
    network.addLayer(new WeightedLayer(3, 2, new Sigmoid(), new HeInitializer(), new RidgeRegression()));
    network.addLayer(new WeightedLayer(2, 1, new Sigmoid(), new HeInitializer(), new RidgeRegression()));

    InterfaceOptimizer* adam = new SGD(3);
    InterfaceLossFunction* mae = new MeanSquarredError();
    InterfaceMetric* accuracy = new Accuracy();

    network.compile(adam, mae, accuracy);

    // Guardar información de las capas
    std::string originalInfo = network.getLayersInfo();

    // Guardar red en archivo binario
    network.saveBinary("test_model.bin");

    // Crear nueva red y cargar desde el archivo binario
    NeuralNetwork loadedNetwork;
    loadedNetwork.loadBinary("test_model.bin");

    // Verificar que la información de las capas sea igual
    std::string loadedInfo = loadedNetwork.getLayersInfo();
    EXPECT_EQ(originalInfo, loadedInfo);

    // Liberar memoria
    delete adam;
    delete mae;
    delete accuracy;
}

TEST(NeuralNetworkTest, TrainAndSave) {
    // Crear red inicial
    NeuralNetwork network;
    network.addLayer(new WeightedLayer(2, 3, new Sigmoid(), new HeInitializer(), new RidgeRegression()));
    network.addLayer(new WeightedLayer(3, 1, new Sigmoid(), new HeInitializer(), new RidgeRegression()));

    InterfaceOptimizer* sgd = new SGD(0.01);
    InterfaceLossFunction* mse = new MeanSquarredError();
    InterfaceMetric* accuracy = new Accuracy();

    network.compile(sgd, mse, accuracy);

    // Crear un dataset simple
    std::vector<std::vector<double>> features = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    std::vector<std::vector<double>> labels = {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };
    DataSet dataset(features, labels);

    // Entrenar la red
    network.fit(dataset, 10, 4, 1);

    // Guardar la red tras el entrenamiento
    network.saveBinary("trained_model.bin");

    // Crear una nueva red y cargar desde el archivo binario
    NeuralNetwork loadedNetwork;
    loadedNetwork.loadBinary("trained_model.bin");

    // Verificar que las capas de ambas redes coincidan tras el entrenamiento
    std::string originalInfo = network.getLayersInfo();
    std::string loadedInfo = loadedNetwork.getLayersInfo();
    EXPECT_EQ(originalInfo, loadedInfo);

    // Liberar memoria
    delete sgd;
    delete mse;
    delete accuracy;
}

TEST(NeuralNetworkTest, SaveAndLoadLayersHeDropout) {
    // Crear red inicial
    NeuralNetwork network;
    network.addLayer(new WeightedLayer(3, 2, new Sigmoid(), new HeInitializer(), new RidgeRegression()));
    network.addLayer(new Dropout(0.5));
    network.addLayer(new WeightedLayer(2, 1, new Sigmoid(), new HeInitializer(), new RidgeRegression()));

    InterfaceOptimizer* adam = new SGD(3);
    InterfaceLossFunction* mae = new MeanSquarredError();
    InterfaceMetric* accuracy = new Accuracy();

    network.compile(adam, mae, accuracy);

    // Guardar información de las capas
    std::string originalInfo = network.getLayersInfo();

    // Guardar red en archivo binario
    network.saveBinary("test_model.bin");

    // Crear nueva red y cargar desde el archivo binario
    NeuralNetwork loadedNetwork;
    loadedNetwork.loadBinary("test_model.bin");

    // Verificar que la información de las capas sea igual
    std::string loadedInfo = loadedNetwork.getLayersInfo();
    EXPECT_EQ(originalInfo, loadedInfo);

    // Liberar memoria
    delete adam;
    delete mae;
    delete accuracy;
}