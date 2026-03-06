#include <gtest/gtest.h>
#include <fstream>
#include <sstream>
#include "../src/NEAT/Genome.h"

namespace {
    constexpr int NUM_INPUTS = 2;
    constexpr int NUM_OUTPUTS = 2;
    constexpr double TEST_FITNESS = 42.0;
    const std::string GENOME_SAVE_FILE = "test_genome.txt";
    const std::string EMPTY_GENOME_SAVE_FILE = "empty_genome.txt";
    const std::vector<double> FORWARD_PASS_INPUT = {0.0, 1.0};
}

TEST(GenomeTest, SaveAndLoad) {
    Config config;
    InnovationCounter innovationCounter;

    Genome originalGenome(config, innovationCounter, NUM_INPUTS, NUM_OUTPUTS);
    originalGenome.addConnection(0, NUM_INPUTS);       // Input 0 -> Output 0
    originalGenome.addConnection(1, NUM_INPUTS + 1);   // Input 1 -> Output 1

    originalGenome.setFitness(TEST_FITNESS);

    originalGenome.save(GENOME_SAVE_FILE);

    Genome loadedGenome(config, innovationCounter);
    loadedGenome.load(GENOME_SAVE_FILE);

    EXPECT_DOUBLE_EQ(originalGenome.getFitness(), loadedGenome.getFitness());

    // Compare forward pass outputs via serialization
    std::ostringstream originalNodes, loadedNodes;
    for (const auto& connection : originalGenome.forwardPass(FORWARD_PASS_INPUT)) {
        originalNodes << connection;
    }
    for (const auto& connection : loadedGenome.forwardPass(FORWARD_PASS_INPUT)) {
        loadedNodes << connection;
    }
    EXPECT_EQ(originalNodes.str(), loadedNodes.str());

    std::remove(GENOME_SAVE_FILE.c_str());
}

TEST(GenomeTest, SaveAndLoadEmptyGenome) {
    Config config;
    InnovationCounter innovationCounter;

    Genome originalGenome(config, innovationCounter);

    originalGenome.save(EMPTY_GENOME_SAVE_FILE);

    Genome loadedGenome(config, innovationCounter);
    loadedGenome.load(EMPTY_GENOME_SAVE_FILE);

    EXPECT_DOUBLE_EQ(originalGenome.getFitness(), loadedGenome.getFitness());

    std::remove(EMPTY_GENOME_SAVE_FILE.c_str());
}
