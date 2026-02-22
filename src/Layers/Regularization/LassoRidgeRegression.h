//
// Created by korone on 1/10/25.
//

#ifndef LASSORIDGEREGRESSION_H
#define LASSORIDGEREGRESSION_H
#include "LassoRegression.h"
#include "RidgeRegression.h"
#include "MathUtils/Matrix.h"

class LassoRidgeRegression : public LassoRegression, public RidgeRegression
{
public:
    LassoRidgeRegression(const double ridgeAlpha = 0.01, const double lassoAlpha = 0.01)
      : LassoRegression(lassoAlpha), RidgeRegression(ridgeAlpha) {}

    double compute(Matrix &weights) const override
    {
        return RidgeRegression::compute(weights) + LassoRegression::compute(weights);
    }

    [[nodiscard]] LassoRidgeRegression* clone() const override {return new LassoRidgeRegression(*this);}

    [[nodiscard]] std::string getType() { return "LassoRidge"; }
};


#endif //LASSORIDGEREGRESSION_H
