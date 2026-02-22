//
// Created by korone on 1/11/25.
//

#ifndef UWU_LEARNER_VECTOR_H
#define UWU_LEARNER_VECTOR_H
#include <functional>
#include <vector>
#include <fstream>
#include <sstream>

#include "Matrix.h"

class Matrix;

namespace uwu
{

    class Vector
    {
    public:
        std::vector<double> data_;

        Vector() = default;
        ~Vector() = default;

        Vector(const std::vector<double> &vector): data_(vector) {};
        Vector(const  Vector& other) : data_(other.data_) {}
        Vector(double n, double initialValue = 0.0);
        [[nodiscard]] const std::vector<double>& getData() const { return data_; }
        Vector& operator=(const  Vector& vector) = default;
        Vector& operator+=(const  Vector& vector);
        Vector& operator*=(const  Vector& vector);
        Vector& operator-=(const  Vector& vector);
        Vector operator-(const  Vector& vector) const;
        Vector operator+(const  Vector& vector) const;
        Vector operator+(double value) const;
        Vector operator*(const double& value) const;
        Vector operator/(const double& value) const;
        Vector operator/(const Vector& vector) const;
        Vector operator^(int pow) const;
        Vector operator-() const;
        void iterate(const std::function<double(double)> &func);
        void fill(double value);
        void saveToBinary(std::ofstream &file) const;
        void loadFromBinary(std::ifstream& file);

        [[nodiscard]] std::size_t size() const
        {
        return data_.size();
        }

        const double& operator[](size_t index) const
        {
        return data_[index];
        }

        double& operator[](size_t index) {
        return data_[index];
        }

        [[nodiscard]] bool empty() const {
        return data_.empty();
        }

        static  Vector dotProduct(const Matrix &m, const  Vector &v);
        void resize(size_t newSize, const double& defaultValue = double());
        [[nodiscard]] std::string toString() const;

        friend class ::Matrix;
        friend bool operator==(const  Vector& lhs, const  Vector& rhs) {
            if (lhs.size() != rhs.size()) {
                return false;
            }
            for (std::size_t i = 0; i < lhs.size(); ++i) {
                if (lhs[i] != rhs[i]) {
                    return false;
                }
            }
            return true;
        }
    };
}


#endif //UWU_LEARNER_VECTOR_H
