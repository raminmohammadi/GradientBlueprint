import numpy as np

class Optimizers:
    @staticmethod
    def SGD(parameters, gradients, learning_rate, multiclass=False):
        """
        Performs stochastic gradient descent optimization.

        Parameters:
            parameters (list): List of parameters (e.g., weights and biases).
            gradients (dict): Dictionary containing gradients for each parameter.
            learning_rate (float): Learning rate for the optimization.
            multiclass (bool, optional): Flag indicating if the problem is multiclass. Defaults to False.
        """
        if multiclass:
            for parameter in parameters:
                for p in parameter:
                    p.data += -learning_rate * np.mean(gradients[p.label])
        else:
            for p in parameters:
                p.data += -learning_rate * np.mean(gradients[p.label])

    @staticmethod
    def batch_gradient_descent(parameters, learning_rate, batch_size, multiclass=False):
        """
        Performs batch gradient descent optimization.

        Parameters:
            parameters (list): List of parameters (e.g., weights and biases).
            learning_rate (float): Learning rate for the optimization.
            batch_size (int): Size of the mini-batch.
            multiclass (bool, optional): Flag indicating if the problem is multiclass. Defaults to False.
        """
        if multiclass:
            for parameter in parameters:
                for p in parameter:
                    p.data += -learning_rate * p.grad / batch_size
        else:
            for p in parameters:
                p.data += -learning_rate * p.grad / batch_size
                p.grad = 0.0
