//
// Created by korone on 1/11/25.
//

#include "Matrix.h"

#include <cmath>
#include <execution>

Matrix::Matrix(int rows, int cols,  double initialValue)
{
    this->data_ = std::vector<std::vector< double>>(rows, std::vector< double>(cols, initialValue));
    this->rows_ = rows;
    this->columns_ = cols;
}

Matrix::Matrix(const Matrix &other)
{
    this->rows_ = other.rows_;
    this->columns_ = other.columns_;

    for (const auto & i : other.data_)
    {
        this->data_.push_back(i);
    }
}

Matrix::Matrix(std::vector<std::vector<double>> matriz)
{
    if (!matriz.empty())
    {
        this->data_ = matriz;
        this->rows_ = matriz.size();
        this->columns_ = matriz[0].size();
    }
}

Matrix& Matrix::operator+=(const double& value)
{
    std::for_each(std::execution::par, this->data_.begin(), this->data_.end(),
                  [&](std::vector<double>& row) {
                      std::transform(std::execution::par, row.begin(), row.end(), row.begin(),
                                     [value](double& v) { return v + value; });
                  });

    return *this;
}

Matrix& Matrix::operator+=(const Matrix& m)
{
    // Parallelized element-wise addition using std::transform for each row
    std::for_each(std::execution::par, this->data_.begin(), this->data_.end(),
                  [&m, idx = 0](std::vector<double>& row) mutable {
                      std::transform(std::execution::par, row.begin(), row.end(), m.data_[idx].begin(), row.begin(),
                                     std::plus<double>());
                      ++idx; // Increment index after processing each row
                  });

    return *this;
}
Matrix& Matrix::operator/=(const int& value)
{
    // Parallelized element-wise division using std::transform for the entire matrix
    std::for_each(std::execution::par, this->data_.begin(), this->data_.end(),
                  [value](std::vector<double>& row) {
                      std::transform(std::execution::par, row.begin(), row.end(), row.begin(),
                                     [value](double& val) { return val / value; });
                  });

    return *this;
}

Matrix & Matrix::operator-=(const Matrix &other)
{
    // Parallelized element-wise subtraction using std::transform for the entire matrix
    std::for_each(std::execution::par, this->data_.begin(), this->data_.end(),
                  [&other, i = 0](std::vector<double>& row) mutable {
                      std::transform(std::execution::par, row.begin(), row.end(), other.data_[i].begin(),
                                     row.begin(), std::minus<double>());
                      ++i;
                  });

    return *this;
}

Matrix Matrix::operator*(const double &value) const
{
    Matrix result(this->rows_, this->columns_, 0.0);

    // Parallelized element-wise multiplication using std::transform for the entire matrix
    std::for_each(std::execution::par, this->data_.begin(), this->data_.end(),
                  [&result, value, i = 0](const std::vector<double>& row) mutable {
                      std::transform(std::execution::par, row.begin(), row.end(), result.data_[i].begin(),
                                     [value](double element) { return value * element; });
                      ++i;
                  });

    return result;
}

Matrix Matrix::operator-(const Matrix &other) const
{
    Matrix result(this->rows_, this->columns_, 0.0);

    // Parallelized element-wise subtraction using std::transform for the entire matrix
    std::for_each(std::execution::par, this->data_.begin(), this->data_.end(),
                  [&result, &other, i = 0](const std::vector<double>& row) mutable {
                      std::transform(std::execution::par, row.begin(), row.end(), other.data_[i].begin(),
                                     result.data_[i].begin(), std::minus<double>());
                      ++i;
                  });

    return result;
}


Matrix Matrix::operator^(const double &value) const
{
    Matrix result(this->rows_, this->columns_, 0.0);

    // Parallelized element-wise multiplication using std::transform for the entire matrix
    std::for_each(std::execution::par, this->data_.begin(), this->data_.end(),
                  [&result, value, i = 0](const std::vector<double>& row) mutable {
                      std::transform(std::execution::par, row.begin(), row.end(), result.data_[i].begin(),
                                     [value](const double element) { return std::pow(element, value); });
                      ++i;
                  });

    return result;
}

Matrix Matrix::operator/(const Matrix &other) const
{
    Matrix result(this->rows_, this->columns_, 0.0);

    // Parallelized element-wise division using std::transform for the entire matrix
    std::for_each(std::execution::par, this->data_.begin(), this->data_.end(),
                  [&result, &other, i = 0](const std::vector<double>& row) mutable {
                      std::transform(std::execution::par, row.begin(), row.end(), other.data_[i].begin(),
                                     result.data_[i].begin(), std::divides<double>());
                      ++i;
                  });

    return result;
}


void Matrix::iterate(const std::function<double(double)> &func)
{
    std::for_each(std::execution::par, this->data_.begin(), this->data_.end(),
              [&](std::vector<double>& row) {
                  std::transform(std::execution::par, row.begin(), row.end(), row.begin(), func);
              });
}

void Matrix::push_back(const uwu::Vector &vector)
{
    this->data_.push_back(vector.data_);
    this->rows_ = this->rows_ + 1;
    this->columns_ = vector.size();
}

void Matrix::fill(double value)
{
    // Parallelized fill using std::fill and std::execution::par
    std::for_each(std::execution::par, this->data_.begin(), this->data_.end(),
                  [value](std::vector<double>& row) {
                      std::fill(std::execution::par, row.begin(), row.end(), value);
                  });
}

void Matrix::fill(std::function<double()> func)
{
    // Parallelized fill using std::for_each and func to generate values
    std::for_each(std::execution::par, this->data_.begin(), this->data_.end(),
                  [&](std::vector<double>& row) {
                      std::for_each(std::execution::par, row.begin(), row.end(), [&](double& v) {
                          v = func();  // Assign value from func() to the element
                      });
                  });
}

Matrix Matrix::operator+(const Matrix &other) const
{
    Matrix result(this->rows_, this->columns_, 0.0);

    // Parallelized element-wise division using std::transform for the entire matrix
    std::for_each(std::execution::par, this->data_.begin(), this->data_.end(),
                  [&result, &other, i = 0](const std::vector<double>& row) mutable {
                      std::transform(std::execution::par, row.begin(), row.end(), other.data_[i].begin(),
                                     result.data_[i].begin(), std::plus<>());
                      ++i;
                  });

    return result;
}

Matrix Matrix::outerProduct(const uwu::Vector &vector1, const uwu::Vector &vector2)
{
    Matrix m(vector1.size(), vector2.size());

    // Parallelized loops for outer product computation
    std::for_each(std::execution::par, m.data_.begin(), m.data_.end(),
                  [&](std::vector<double>& row) {
                      size_t i = &row - &m.data_[0];  // Get the row index
                      std::transform(std::execution::par, vector2.data_.begin(), vector2.data_.end(), row.begin(),
                                     [&](double value) { return vector1.data_[i] * value; });
                  });

    return m;
}

Matrix Matrix::transpose() const
{
    Matrix m(this->columns_, this->rows_);

    // Parallelized loops for matrix transposition
    std::for_each(std::execution::par, m.data_.begin(), m.data_.end(),
                  [&](std::vector<double>& row) {
                      size_t rowIndex = &row - &m.data_[0];  // Get the row index
                      std::transform(std::execution::par, this->data_.begin(), this->data_.end(), row.begin(),
                                     [&](const std::vector<double>& srcRow) {
                                         return srcRow[rowIndex];  // Transpose by column access
                                     });
                  });

    return m;
}

void Matrix::saveToBinary(std::ofstream &file) const {
    // Guardar el número de filas y columnas
    file.write(reinterpret_cast<const char*>(&rows_), sizeof(rows_));
    file.write(reinterpret_cast<const char*>(&columns_), sizeof(columns_));

    // Guardar todos los valores de la matriz
    for (const auto &row : data_) {
        file.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));
    }
}

void Matrix::loadFromBinary(std::ifstream &file) {
    // Leer el número de filas y columnas
    file.read(reinterpret_cast<char*>(&rows_), sizeof(rows_));
    file.read(reinterpret_cast<char*>(&columns_), sizeof(columns_));

    // Redimensionar y leer los datos
    data_.resize(rows_, std::vector<double>(columns_));
    for (auto &row : data_) {
        file.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
    }
}

std::string Matrix::toString() const {
    std::ostringstream oss;
    for (size_t i = 0; i < rows(); ++i) {
        for (size_t j = 0; j < columns(); ++j) {
            oss << (*this)(i, j) << " ";  // Suponiendo que tienes operador `()` para acceso
        }
        oss << "\n";
    }
    return oss.str();
}
