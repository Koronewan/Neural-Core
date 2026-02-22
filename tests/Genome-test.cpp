#include <gtest/gtest.h>
#include <fstream>
#include <sstream>
#include "../src/NEAT/Genome.h"

TEST(GenomeTest, SaveAndLoad) {
    // Crear configuraciones e instancias necesarias
    Config config;
    InnovationCounter innovationCounter;

    // Crear un genome original con nodos y conexiones
    Genome originalGenome(config, innovationCounter, 2, 2); // 2 entradas, 2 salidas
    originalGenome.addConnection(0, 2);
    originalGenome.addConnection(1, 3);

    // Establecer un valor de fitness
    originalGenome.setFitness(42.0);

    // Guardar el genome en un archivo
    std::string fileName = "test_genome.txt";
    originalGenome.save(fileName);

    // Cargar el genome desde el archivo
    Genome loadedGenome(config, innovationCounter);
    loadedGenome.load(fileName);

    // Verificar atributos importantes
    EXPECT_DOUBLE_EQ(originalGenome.getFitness(), loadedGenome.getFitness());

    // Serializar nodos y conexiones para comparar sus datos
    std::ostringstream originalNodes, loadedNodes;
    for (const auto& connection : originalGenome.forwardPass({0.0, 1.0})) {
        originalNodes << connection;
    }
    for (const auto& connection : loadedGenome.forwardPass({0.0, 1.0})) {
        loadedNodes << connection;
    }
    EXPECT_EQ(originalNodes.str(), loadedNodes.str());

    // Eliminar el archivo después de la prueba
    std::remove(fileName.c_str());
}

TEST(GenomeTest, SaveAndLoadEmptyGenome) {
    // Crear configuraciones e instancias necesarias
    Config config;
    InnovationCounter innovationCounter;

    // Crear un genome vacío
    Genome originalGenome(config, innovationCounter);

    // Guardar el genome en un archivo
    std::string fileName = "empty_genome.txt";
    originalGenome.save(fileName);

    // Cargar el genome desde el archivo
    Genome loadedGenome(config, innovationCounter);
    loadedGenome.load(fileName);

    // Verificar que los atributos sean consistentes
    EXPECT_DOUBLE_EQ(originalGenome.getFitness(), loadedGenome.getFitness());

    // Eliminar el archivo después de la prueba
    std::remove(fileName.c_str());
}
