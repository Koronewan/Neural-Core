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

//
// Created by tar87 on 20/12/2024.
//

int main(int argc, char *argv[]) {

    NeuralNetwork network;
    network.addLayer(new WeightedLayer(748, 30,new Sigmoid(), new GlorotInitializer(), new RidgeRegression()));
    network.addLayer(new WeightedLayer(30, 10,new Sigmoid(), new GlorotInitializer(), new RidgeRegression()));

    InterfaceOptimizer* adam = new SGD(3);
    InterfaceLossFunction* mae = new MeanSquarredError();
    InterfaceMetric* accuracy = new Accuracy();

    network.compile(adam,mae,accuracy);

    MNISTLoader mnistLoader;
    if (!mnistLoader.load("/home/korone/uwu_gang_ra24/proyecto2/src/data")) {
        std::cerr << "Failed to load MNIST dataset!" << std::endl;
        return 1;
    }

    std::vector<std::vector<double>> inputs = mnistLoader.getTrainingImages();
    std::vector<std::vector<double>> labels = mnistLoader.getTrainingLabels();

    DataSet data = DataSet(inputs, labels);

    data.shuffle();
    network.fit(data,30,10,0,EarlyStopping());

    delete adam;
    delete mae;
    delete accuracy;

}
