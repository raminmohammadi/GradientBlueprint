import sys, os
# Get the parent directory
parent_directory = sys.path[0]  # Assumes the script is in the parent directory

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(parent_directory)))

from src.Helpers.DrawGraph import draw_dot
from src.Gradient.Gradient import Variable
from src.Optimizers.optimizers import Optimizers
from src.Cost_functions.Cost_functions import CostFunction
import random
import numpy as np

class LinearRegression:
    def __init__(self, input_dim):
        """
        Initializes the Linear Regression model.

        Parameters:
            input_dim (int): Dimensionality of the input data.
        """
        self.w = [Variable(random.uniform(-1, 1), label=f"W{_}") for _ in range(input_dim)]
        self.b = Variable(random.uniform(-1, 1), label="b")
        self.costFunction = getattr(CostFunction, 'sse')
        
    def parameters(self):
        """
        Get the parameters (weights and bias) of the model.

        Returns:
            list: List of parameters.
        """
        return self.w + [self.b]

    def predict(self, X):
        """
        Predicts the output for a given input.

        Parameters:
            X (array_like): Input data.

        Returns:
            list: Predicted outputs.
        """
        ypreds = [self(x) for x in X]
        return ypreds
    
    def __call__(self, x):
        """
        Computes the output of the model for a given input.

        Parameters:
            x (array_like): Input data.

        Returns:
            float: Output of the model.
        """
        y_hat = sum((w_i * x_i for w_i, x_i in zip(self.w, x.T)), self.b)
        return y_hat
    
    def __draw__(self, x):
        """
        Draw the computation graph of the model.

        Parameters:
            x (array_like): Input data.

        Returns:
            str: Graph in DOT format.
        """
        return draw_dot(self.__call__(x))
     
    def regularizer(self, regularization_term):
        """
        Computes the regularization term.

        Parameters:
            regularization_term (float): Regularization term.

        Returns:
            Variable: Regularization term.
        """
        penalty_term = Variable(0, label='regularization_term')
        for param in self.w:
            penalty_term += (param ** 2)
        return penalty_term * regularization_term / (2 * len(self.w))
    
    def fit(self, X, y, learning_rate=0.001, num_epochs=300,
            batch_size=1024, optimizer='SGD', regularization_term=0.05):
        """
        Fits the Linear Regression model to the given data.

        Parameters:
            X (array_like): Input data.
            y (array_like): Target values.
            learning_rate (float, optional): Learning rate for optimization. Defaults to 0.001.
            num_epochs (int, optional): Number of epochs for training. Defaults to 300.
            batch_size (int, optional): Batch size for mini-batch optimization. Defaults to 1024.
            optimizer (str, optional): Optimization algorithm ('SGD' or 'batch_gradient_descent'). Defaults to 'SGD'.
            regularization_term (float, optional): Regularization term. Defaults to 0.05.
        """
        for epoch in range(num_epochs):
            if batch_size is None:
                Xb, yb = X, y
            else:
                ri = np.random.permutation(X.shape[0])[:batch_size]
                Xb, yb = X[ri], y[ri]

            ypreds = self.predict(Xb)      

            # Forward pass
            if optimizer == 'SGD':
                losses = [self.costFunction(y_hat, y_) + self.regularizer(regularization_term) for y_hat, y_ in zip(ypreds, yb)]
                # Backward Pass
                gradients = {parameter.label: [] for parameter in self.parameters()}
                for k, loss in enumerate(losses):
                    loss.backward()
                    self.loss = loss
                    for parameter in self.parameters():
                        gradients[parameter.label].append(parameter.grad)
                        parameter.grad = 0.0

            else:
                # batch GD
                loss = sum((self.costFunction(y_hat, y_)  for ypred, y_ in zip(ypreds, yb)), Variable(0)) + self.regularizer(regularization_term)
                loss.backward()

            # Update using specified optimizer
            if optimizer == 'SGD':
                Optimizers.SGD(self.parameters(), gradients, learning_rate)
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}, Loss: {sum([loss.data for loss in losses])}")

            elif optimizer == 'batch_gradient_descent':
                Optimizers.batch_gradient_descent(self.parameters(), learning_rate, batch_size)
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}, Loss: {loss.data}")
