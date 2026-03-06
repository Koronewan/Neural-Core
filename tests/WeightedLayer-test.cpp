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
