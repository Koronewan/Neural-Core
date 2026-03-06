//
// Created by root on 1/16/25.
//

#include <gtest/gtest.h>
#include "EarlyStopping.h"

namespace {
    constexpr int DEFAULT_PATIENCE = 2;
    constexpr int HIGHER_PATIENCE = 3;
    constexpr double SMALL_DELTA = 0.01;
    constexpr double LARGER_DELTA = 0.05;
    const std::string MINIMIZE_MODE = "min";
    const std::string MAXIMIZE_MODE = "max";
}

TEST(EarlyStoppingTest, InitializeDefaultConstructor) {
    EarlyStopping earlyStopping;

    EXPECT_EQ(earlyStopping.shouldStop(), false);
    EXPECT_NO_THROW(earlyStopping.reset());
}

TEST(EarlyStoppingTest, InitializeWithParameters) {
    EarlyStopping earlyStopping(HIGHER_PATIENCE, SMALL_DELTA, MINIMIZE_MODE);

    EXPECT_EQ(earlyStopping.shouldStop(), false);
    EXPECT_NO_THROW(earlyStopping.reset());
}

TEST(EarlyStoppingTest, MinimizeMetric) {
    EarlyStopping earlyStopping(DEFAULT_PATIENCE, SMALL_DELTA, MINIMIZE_MODE);

    // Simulate improving metrics (decreasing loss)
    earlyStopping.evaluate(0.5);
    EXPECT_FALSE(earlyStopping.shouldStop());

    earlyStopping.evaluate(0.4);
    EXPECT_FALSE(earlyStopping.shouldStop());

    // Simulate stagnating/worsening metrics
    earlyStopping.evaluate(0.41);
    earlyStopping.evaluate(0.42);
    earlyStopping.evaluate(0.43);
    EXPECT_TRUE(earlyStopping.shouldStop());
}

TEST(EarlyStoppingTest, MaximizeMetric) {
    EarlyStopping earlyStopping(DEFAULT_PATIENCE, SMALL_DELTA, MAXIMIZE_MODE);

    // Simulate improving metrics (increasing accuracy)
    earlyStopping.evaluate(0.7);
    EXPECT_FALSE(earlyStopping.shouldStop());

    earlyStopping.evaluate(0.72);
    EXPECT_FALSE(earlyStopping.shouldStop());

    // Simulate stagnating/worsening metrics
    earlyStopping.evaluate(0.71);
    earlyStopping.evaluate(0.70);
    earlyStopping.evaluate(0.70);
    EXPECT_TRUE(earlyStopping.shouldStop());
}

TEST(EarlyStoppingTest, ResetMethod) {
    EarlyStopping earlyStopping(DEFAULT_PATIENCE, SMALL_DELTA, MINIMIZE_MODE);

    // Trigger early stopping with worsening metrics
    earlyStopping.evaluate(0.5);
    earlyStopping.evaluate(0.6);
    earlyStopping.evaluate(0.7);
    earlyStopping.evaluate(0.8);
    EXPECT_TRUE(earlyStopping.shouldStop());

    // Reset and verify it restarts correctly
    earlyStopping.reset();
    EXPECT_FALSE(earlyStopping.shouldStop());

    earlyStopping.evaluate(0.4);
    EXPECT_FALSE(earlyStopping.shouldStop());
}

TEST(EarlyStoppingTest, EvaluateWithDelta) {
    EarlyStopping earlyStopping(HIGHER_PATIENCE, LARGER_DELTA, MINIMIZE_MODE);

    // Small improvements below the delta threshold
    earlyStopping.evaluate(0.5);
    earlyStopping.evaluate(0.49); // Not a significant improvement (< delta)
    earlyStopping.evaluate(0.48);
    EXPECT_FALSE(earlyStopping.shouldStop());

    // No sufficient improvement; should trigger stop
    earlyStopping.evaluate(0.48);
    earlyStopping.evaluate(0.48);
    earlyStopping.evaluate(0.48);
    EXPECT_TRUE(earlyStopping.shouldStop());
}

