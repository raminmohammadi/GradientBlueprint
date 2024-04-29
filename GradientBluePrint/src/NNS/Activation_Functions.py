import sys

# Get the parent directory
parent_directory = sys.path[0]  # Assumes the script is in the parent directory

# Add the parent directory to the Python path
sys.path.append(parent_directory)
from src.Gradient.Gradient import Variable

import math

class Activations(Variable):
    """
    A class containing static methods for various activation functions.

    Inherits from Variable class for backward propagation.
    """

    @staticmethod
    def sigmoid(self):
        """
        Computes the sigmoid activation function.

        Returns:
            Variable: Output variable after applying the sigmoid function.
        """
        n = self.data
        t = 1 / (1 + math.exp(-n))
        output = Variable(t, _children=(self, ), _op='sigmoid')

        def _backward():
            self.grad += t * (1 - t) * output.grad

        output._backward = _backward
        return output

    @staticmethod
    def tanh(self):
        """
        Computes the hyperbolic tangent activation function.

        Returns:
            Variable: Output variable after applying the tanh function.
        """
        n = self.data
        t = (math.exp(2 * n) - 1) / (math.exp(2 * n) + 1)
        output = Variable(t, _children=(self, ), _op='tanh')

        def _backward():
            self.grad += (1 - t ** 2) * output.grad

        output._backward = _backward
        return output

    @staticmethod
    def relu(self):
        """
        Computes the rectified linear unit (ReLU) activation function.

        Returns:
            Variable: Output variable after applying the ReLU function.
        """
        n = self.data
        t = n if n > 0 else 0
        output = Variable(t, _children=(self, ), _op='relu')

        def _backward():
            if t > 0:
                self.grad += 1 * output.grad
            else:
                self.grad += 0 * output.grad

        output._backward = _backward
        return output

    @staticmethod
    def leaky_relu(self):
        """
        Computes the leaky ReLU activation function.

        Returns:
            Variable: Output variable after applying the leaky ReLU function.
        """
        n = self.data
        t = n if n > 0 else 0.01 * n
        output = Variable(t, _children=(self, ), _op='leaky relu')

        def _backward():
            if t > 0:
                self.grad += 1 * output.grad
            else:
                self.grad += 0.01 * output.grad

        output._backward = _backward
        return output

    @staticmethod
    def softmax(self):
        """
        Computes the softmax activation function.

        Returns:
            list: List of output variables after applying the softmax function.
        """
        n = [x.data for x in self]
        exp_n = [math.exp(x) for x in n]
        sum_exp_n = sum(exp_n)
        t = [x / sum_exp_n for x in exp_n]
        output = [Variable(t[i], _children=(self[i], ), _op='softmax') for i in range(len(t))]

        def _backward():
            for i in range(len(self)):
                for j in range(len(self)):
                    if i == j:
                        self[i].grad += t[i] * (1 - t[i]) * output[j].grad
                    else:
                        self[i].grad += -t[i] * t[j] * output[j].grad

        for i in range(len(output)):
            output[i]._backward = _backward

        return output
