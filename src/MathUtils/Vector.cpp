//
// Created by korone on 1/11/25.
//
#include "Vector.h"

#include <cmath>
#include <numeric>  // for std::transform_reduce
#include <execution> // for std::execution::par


uwu::Vector::Vector(double n, double initialValue)
{
    this->data_ = std::vector(n, initialValue);
}

uwu::Vector& uwu::Vector::operator+=(const uwu::Vector& vector)
{
    std::transform(std::execution::par, this->data_.begin(), this->data_.end(), vector.data_.begin(), this->data_.begin(),
                   [](double a, double b) { return a + b; });
    return *this;
}

uwu::Vector& uwu::Vector::operator*=(const uwu::Vector& vector)
{
    std::transform(std::execution::par, this->data_.begin(), this->data_.end(), vector.data_.begin(), this->data_.begin(),
                   [](double a, double b) { return a * b; });
    return *this;
}

uwu::Vector& uwu::Vector::operator-=(const Vector& vector)
{
    std::transform(std::execution::par, this->data_.begin(), this->data_.end(), vector.data_.begin(), this->data_.begin(),
                   [](double a, double b) { return a - b; });
    return *this;
}

uwu::Vector uwu::Vector::operator-(const uwu::Vector& vector) const
{
    uwu::Vector result(this->size(), 0);
    std::transform(std::execution::par, this->data_.begin(), this->data_.end(), vector.data_.begin(), result.data_.begin(),
                   [](double a, double b) { return a - b; });
    return result;
}

uwu::Vector uwu::Vector::operator+(double value) const
{
    uwu::Vector result(this->size(), 0);
    std::transform(std::execution::par, this->data_.begin(), this->data_.end(), result.data_.begin(),
                   [value](double elem) { return elem + value; });
    return result;
}

uwu::Vector uwu::Vector::operator+(const uwu::Vector& vector) const
{
    uwu::Vector result(this->size(), 0);
    std::transform(std::execution::par, this->data_.begin(), this->data_.end(), vector.data_.begin(), result.data_.begin(),
                   [](double a, double b) { return a + b; });
    return result;
}

uwu::Vector uwu::Vector::operator*(const double &value) const
{
    uwu::Vector result(this->size(), 0);
    std::transform(std::execution::par, this->data_.begin(), this->data_.end(), result.data_.begin(),
                   [value](double elem) { return elem * value; });
    return result;
}

uwu::Vector uwu::Vector::operator/(const double &value) const
{
    uwu::Vector result(this->size(), 0);
    std::transform(std::execution::par, this->data_.begin(), this->data_.end(), result.data_.begin(),
                   [value](double elem) { return elem / value; });
    return result;
}

uwu::Vector uwu::Vector::operator/(const Vector &vector) const
{
    uwu::Vector result(this->size(), 0);
    std::transform(std::execution::par, this->data_.begin(), this->data_.end(), vector.data_.begin(), result.data_.begin(),
                   [](double elem, double vecElem) { return elem / vecElem; });
    return result;
}

uwu::Vector uwu::Vector::operator^(int pow) const
{
    uwu::Vector result(this->size(), 0);
    std::transform(std::execution::par, this->data_.begin(), this->data_.end(), result.data_.begin(),
                   [pow](double value) { return std::pow(value, pow); });
    return result;
}

uwu::Vector uwu::Vector::operator-() const
{
    uwu::Vector result(this->size(), 0);
    std::transform(std::execution::par, this->data_.begin(), this->data_.end(), result.data_.begin(),
                   [](double value) { return -value; });
    return result;
}


void uwu::Vector::iterate(const std::function<double(double)> &func)
{
    std::transform(std::execution::par, this->data_.begin(),
    this->data_.end(), this->data_.begin(), func);
}

void uwu::Vector::fill(double value)
{
    std::fill(this->data_.begin(), this->data_.end(), value);
}

uwu::Vector uwu::Vector::dotProduct(const Matrix &m, const uwu::Vector &v)
{
    std::vector<double> result(m.rows(), 0.0);

    std::transform(std::execution::par, m.data_.begin(), m.data_.end(), result.begin(),
        [&v](const std::vector<double> &row) {
            return std::transform_reduce(
                std::execution::par, row.begin(), row.end(), v.data_.begin(), 0.0);
        });

    return Vector(result);
}

void uwu::Vector::resize(size_t newSize, const double &defaultValue)
{
    data_.resize(newSize, defaultValue);
}

void uwu::Vector::saveToBinary(std::ofstream &file) const {
    // Usar uint64_t para asegurar portabilidad
    std::size_t size = data_.size();
    file.write(reinterpret_cast<const char*>(&size), sizeof(size));

    // Guardar los datos del vector
    file.write(reinterpret_cast<const char*>(data_.data()), size * sizeof(double));
}

void uwu::Vector::loadFromBinary(std::ifstream &file) {
    // Leer el tamaño del vector
    std::size_t size;
    file.read(reinterpret_cast<char*>(&size), sizeof(size));

    // Redimensionar el vector y leer los datos
    data_.resize(size);
    file.read(reinterpret_cast<char*>(data_.data()), size * sizeof(double));
}

std::string uwu::Vector::toString() const {
    std::ostringstream oss;
    for (size_t i = 0; i < size(); ++i) {
        oss << (*this)[i] << " ";  // Suponiendo que tienes operador `[]` para acceso
    }
    return oss.str();
}