//
// Created by root on 1/16/25.
//

#include <gtest/gtest.h>
#include "EarlyStopping.h"

TEST(EarlyStoppingTest, InitializeDefaultConstructor) {
    EarlyStopping earlyStopping;

    // Comprobar valores iniciales del constructor por defecto
    EXPECT_EQ(earlyStopping.shouldStop(), false);
    EXPECT_NO_THROW(earlyStopping.reset());
}

TEST(EarlyStoppingTest, InitializeWithParameters) {
    EarlyStopping earlyStopping(3, 0.01, "min");

    // Comprobar valores iniciales
    EXPECT_EQ(earlyStopping.shouldStop(), false);
    EXPECT_NO_THROW(earlyStopping.reset());
}

TEST(EarlyStoppingTest, MinimizeMetric) {
    EarlyStopping earlyStopping(2, 0.01, "min");

    // Simular métricas que mejoran
    earlyStopping.evaluate(0.5);
    EXPECT_FALSE(earlyStopping.shouldStop());

    earlyStopping.evaluate(0.4);
    EXPECT_FALSE(earlyStopping.shouldStop());

    // Simular métricas que no mejoran
    earlyStopping.evaluate(0.41);
    earlyStopping.evaluate(0.42);
    earlyStopping.evaluate(0.43);
    EXPECT_TRUE(earlyStopping.shouldStop());
}

TEST(EarlyStoppingTest, MaximizeMetric) {
    EarlyStopping earlyStopping(2, 0.01, "max");

    // Simular métricas que mejoran
    earlyStopping.evaluate(0.7);
    EXPECT_FALSE(earlyStopping.shouldStop());

    earlyStopping.evaluate(0.72);
    EXPECT_FALSE(earlyStopping.shouldStop());

    // Simular métricas que no mejoran
    earlyStopping.evaluate(0.71);
    earlyStopping.evaluate(0.70);
    earlyStopping.evaluate(0.70);
    EXPECT_TRUE(earlyStopping.shouldStop());
}

TEST(EarlyStoppingTest, ResetMethod) {
    EarlyStopping earlyStopping(2, 0.01, "min");

    // Simular métricas que no mejoran
    earlyStopping.evaluate(0.5);
    earlyStopping.evaluate(0.6);
    earlyStopping.evaluate(0.7);
    earlyStopping.evaluate(0.8);
    EXPECT_TRUE(earlyStopping.shouldStop());

    // Resetear y comprobar que se reinicia correctamente
    earlyStopping.reset();
    EXPECT_FALSE(earlyStopping.shouldStop());

    earlyStopping.evaluate(0.4);
    EXPECT_FALSE(earlyStopping.shouldStop());
}

TEST(EarlyStoppingTest, EvaluateWithDelta) {
    EarlyStopping earlyStopping(3, 0.05, "min");

    // Simular pequeñas mejoras por debajo del delta
    earlyStopping.evaluate(0.5);
    earlyStopping.evaluate(0.49); // No cuenta como mejora significativa
    earlyStopping.evaluate(0.48);
    EXPECT_FALSE(earlyStopping.shouldStop());

    // Sin mejora suficiente, debe detenerse
    earlyStopping.evaluate(0.48);
    earlyStopping.evaluate(0.48);
    earlyStopping.evaluate(0.48);
    EXPECT_TRUE(earlyStopping.shouldStop());
}

