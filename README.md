[![pytest](https://github.com/raminmohammadi/gradientblueprint/actions/workflows/pytest.yml/badge.svg)](https://github.com/raminmohammadi/gradientblueprint/actions/workflows/pytest.yml)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

# Gradient-Based Learning

A comprehensive Python library for gradient-based learning, including neural networks, logistic regression, support vector machines (SVM), and more.

## Introduction

The Gradient-Based Learning Library is a versatile toolset for implementing and experimenting with various machine learning algorithms that rely on gradient descent optimization. From basic linear regression to complex neural networks, this library provides a unified interface for building, training, and evaluating models.

## Features

- Implementation of the Variable class for automatic differentiation
- Support for various activation functions and cost functions
- Modular neural network architecture with customizable layers and activations
- Logistic regression and support vector machine (SVM) implementations
- Optimizers including stochastic gradient descent (SGD) and batch gradient descent
- Extensible and easy-to-use API for adding custom models and functionalities

## Installation

Install the library using pip:

```bash
pip install gradientblueprint
```

## Usage
To use the library, import the necessary modules and classes into your Python code:

```python
from gradientblueprint import Variable, MLP, LogisticRegression, SVM, Optimizer, CostFunction
```

Then, create instances of the provided classes and customize them according to your requirements. Here's a basic example of building and training a neural network:

```python
# Define input dimension and layer dimensions
input_dim = 10
layer_dims = [32, 16, 8]

# Create a multilayer perceptron (MLP)
mlp = MLP(input_dim=input_dim, layers_dim=layer_dims, activations=['relu', 'relu', 'sigmoid'])

# Define optimizer and cost function
optimizer = Optimizer()
cost_function = CostFunction.mean_squared_error

# Train the MLP
for epoch in range(num_epochs):
    # Perform forward pass
    # Perform backward pass and update weights using optimizer

```

Refer to the documentation and examples for detailed usage instructions and customization options.

## Examples
Explore the examples directory for detailed usage examples and tutorials on how to use different components of the library:

- [Simple MLP Example:](GradientBluePrint/src/NNS/) Basic usage example of building and training a multilayer perceptron (MLP).
- [Logistic Regression Example:](/GradientBluePrint/src/Regression/) Example demonstrating how to use logistic regression for binary classification.
- [SVM Example:](/GradientBluePrint/src/SVM/) Example illustrating how to use support vector machines (SVM) for classification tasks.

## Documentation
For detailed [documentation](GradientBluePrint/documentation/), including API reference and usage guidelines, refer to the Documentation.

## Contributing
Contributions to the Gradient-Based Learning Library are welcome! If you find any issues or have suggestions for improvement, please submit a pull request or open an issue on GitHub.

Please read the Contributing Guide for details on how to contribute to this project.

## License
This project is licensed under the MIT License.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=raminmohammadi/GradientBlueprint&type=Date)](https://www.star-history.com/#raminmohammadi/GradientBlueprint&Date)
