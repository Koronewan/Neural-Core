//
// Created by korone on 1/11/25.
//

#ifndef UWU_LEARNER_MATRIX_H
#define UWU_LEARNER_MATRIX_H

#include "Vector.h"

namespace uwu {
    class Vector;
}

class Matrix
{
public:
    int rows_ = 0, columns_ = 0;
    std::vector<std::vector<double>> data_;
public:
    Matrix() = default;
    explicit Matrix(std::vector<std::vector<double>> data);
    Matrix(int rows, int cols,  double initialValue = 0);
    Matrix(const Matrix &other);
    Matrix& operator=(const Matrix& m) = default;
    Matrix& operator+=(const  double& value);
    Matrix& operator+=(const Matrix& other);
    Matrix& operator/=(const int& value);
    Matrix& operator-=(const Matrix& other);
    Matrix operator*(const double &value) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator^(const double &value) const;
    Matrix operator/(const Matrix& other) const;
    void iterate(const std::function<double(double)> &func);
    void saveToBinary(std::ofstream &file) const;
    void loadFromBinary(std::ifstream &file);

    void push_back(const uwu::Vector& vector);
    void fill( double value);
    void fill(std::function< double()> func);

    [[nodiscard]] Matrix transpose() const;

    [[nodiscard]] int rows() const { return rows_; }
    [[nodiscard]] int columns() const { return columns_; }
    const std::vector< double>& operator[](const size_t index) const
    {
        return data_[index];
    }

    void iterate(const std::function<void( double&, int, int)>& func)
    {
        for (int i = 0; i < rows_; i++)
        {
            for (int j = 0; j < columns_; j++)
            {
                func(data_[i][j], i, j); // Modifica directamente data_
            }
        }
    }

    Matrix iterate(const std::function<void( double&, int, int)>& func) const {
        std::vector<std::vector< double>> newData = this->data_;
        for (int i = 0; i < rows_; i++)
        {
            for (int j = 0; j < columns_; j++)
            {
                func(newData[i][j], i, j);
            }
        }
        return Matrix(newData);
    }

    const  double& operator()(int row, int col) const {
        return data_[row][col];
    }

     double& operator()(int row, int col) {
        return data_[row][col];
    }

    [[nodiscard]] int size() const {
        return rows_;
    }

    Matrix operator+(const Matrix & other) const;

    static Matrix outerProduct(const uwu::Vector& vector1, const uwu::Vector& vector2);

    friend bool operator==(const Matrix& lhs, const Matrix& rhs) {
        if (lhs.rows() != rhs.rows() || lhs.columns() != rhs.columns()) {
            return false;
        }
        for (int i = 0; i < lhs.rows(); ++i) {
            for (int j = 0; j < lhs.columns(); ++j) {
                if (lhs(i, j) != rhs(i, j)) {
                    return false;
                }
            }
        }
        return true;
    }
    [[nodiscard]] std::string toString() const;

    friend class uwu::Vector;
};

#endif //UWU_LEARNER_MATRIX_H
