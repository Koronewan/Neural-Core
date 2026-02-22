# C++ Neuro-Core Framework

A high-performance, dependency-free C++ framework for Deep Learning and Evolutionary Computing. This project implements core neural network architectures, backpropagation, and genetic optimization algorithms from scratch.

---

## Technical Overview

* **Language:** Modern C++.
* **Core Paradigm:** Object-Oriented Programming (OOP) with a modular layer-based architecture.
* **Optimization Techniques:** Custom implementations of Backpropagation and Neuroevolution.
* **Dependencies:** None (Pure Standard Library).

---

## Key Features & Engineering Challenges

### 1. Neural Engine from Scratch
Developed a complete neural network lifecycle without high-level libraries like TensorFlow.
* **Layer Abstraction:** Implemented a flexible architecture supporting custom layer depth and neuron density. Includes specialized layers such as Dropout to mitigate overfitting.
* **Activation Functions:** Manual derivation and implementation of Sigmoid, ReLU, and Tanh, along with their respective derivatives for gradient descent.
* **Weight Initialization:** Multiple initialization strategies including **Glorot (Xavier)**, **He**, and **One**.
* **Regularization:** Support for **L1 (Lasso)** and **L2 (Ridge)** regularization to improve model generalization.

### 2. Optimization & Training Strategies
The framework provides several tools to streamline the optimization process:
* **Backpropagation:** A rigorous implementation of the chain rule to minimize Loss Functions (MSE/Cross-Entropy).
* **Advanced Optimizers:** Support for **Adam**, **RMSProp**, and **Stochastic Gradient Descent (SGD)**.
* **Training Management:** Includes standard training loops and **Early Stopping** strategies to save computational resources when metrics converge.

### 3. Scalable OOP Architecture
The framework is designed to be fully extensible through modern OOP patterns:
* **Interface-Driven Design:** Every component (layers, optimizers, initializers) is implemented via abstract interfaces, allowing for seamless addition of new features.
* **Event-Driven System:** A built-in event system monitors the training lifecycle with hooks for:
    - `BatchStart` / `BatchEnd`
    - `FitStart`
    - `EpochStart` / `EpochEnd`

### 4. Neuroevolution (NEAT)
In addition to gradient-based learning, the framework includes an implementation of the **NEAT (NeuroEvolution of Augmenting Topologies)** algorithm for evolving network structures.

---

## Project Structure

* `/src`: Core C++ implementation.
* `/tests`: Unit and integration tests for validating mathematical correctness.

---

## Team & Credits

This project was developed by the **UwU Gang**:

* **Marcos Alemany Manzanaro** (Lead Developer)
* **Nur del Amo** (Developer)
* **Antonio Perea** (Developer)

---

## Usage

1. **Compilation:**
   ```bash
   mkdir build && cd build
   cmake ..
   make
