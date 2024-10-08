import random
import numpy as np
from ..Gradient.Gradient import Variable
from ..Optimizers.optimizers import Optimizers
from ..Cost_functions.Cost_functions import CostFunction


class SVM:
    """
    Support Vector Machine (SVM) classifier.

    Parameters:
    -----------
    input_dim : int
        Dimensionality of the input features.

    Attributes:
    -----------
    alpha : list of Variable objects
        Lagrange multipliers for support vectors.
    b : Variable
        Bias term.

    Methods:
    -----------
    parameters() -> list of Variable objects:
        Get the parameters of the SVM model.
    hinge_loss(y_true, y_pred) -> Variable:
        Compute the hinge loss between true labels and predicted labels.
    fit(X, y):
        Train the SVM model using the given training data and labels.
    compute_gradient(X, y, i) -> dict:
        Compute the gradient of the Lagrangian with respect to parameters.
    update_parameters(gradient, learning_rate):
        Update parameters using gradient descent.
    __repr__() -> str:
        String representation of the SVM model.
    """

    def __init__(self, input_dim, learning_rate = 0.001):
        """
        Initialize the SVM classifier.

        Parameters:
        -----------
        input_dim : int
            Dimensionality of the input features.
        """
        self.w = [Variable(random.uniform(-1, 1), label=f"W{_}") for _ in range(input_dim)]
        self.b = Variable(random.uniform(-1, 1), label="b")
        self.margin = 1.0  # Margin parameter for SVM
        self.costFunction = CostFunction.hinge_loss
        self.alpha = [Variable(0) for _ in range(input_dim)]
        self.learning_rate = learning_rate

    def predict(self, X):
        """
        Predicts the output for a given input x using the SVM model.
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
        y_hat = sum((w_i * x_i for w_i, x_i in zip(self.w, x.T)), self.b)  # Forward pass
        return y_hat
    
    def parameters(self):
        """
        Get the parameters of the SVM model.

        Returns:
        -----------
        list of Variable objects:
            Parameters of the SVM model (alpha and b).
        """
        return self.alpha + self.w + [self.b]

    def fit(self, X, y, num_epochs=5,
            batch_size=1024, optimizer='SGD'):
        """
        Fits the SVM model to the given data.

        Parameters:
            X (array_like): Input data.
            y (array_like): Target values (-1 or 1 for binary classification).
            num_epochs (int, optional): Number of epochs for training. Defaults to 300.
            batch_size (int, optional): Batch size for mini-batch optimization. Defaults to 1024.
            optimizer (str, optional): Optimization algorithm ('SGD' or 'BGD'). Defaults to 'SGD'.
        """
        for epoch in range(num_epochs):
            if batch_size is None:
                Xb, yb = X, y
            else:
                ri = np.random.permutation(X.shape[0])[:batch_size]
                Xb, yb = X[ri], y[ri]

            ypreds = self.predict(Xb)    

            # Forward pass
            losses = [self.costFunction(ypred, y_) for ypred, y_ in zip(ypreds, yb)]
            gradients = {parameter.label: [] for parameter in self.parameters()}
                        
            for k, loss in enumerate(losses):
                loss.backward()
                self.loss = loss
                for parameter in self.parameters():
                    gradients[parameter.label].append(parameter.grad)
                    parameter.grad = 0.0

            # Update using specified optimizer
            if optimizer == 'SGD':
                Optimizers.SGD(self.parameters(), gradients, self.learning_rate)
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}, Loss: {sum([loss.data for loss in losses])}")

            elif optimizer == 'BGD':
                Optimizers.batch_gradient_descent(self.parameters(), self.learning_rate, batch_size)
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}, Loss: {sum([loss.data for loss in losses])}")

    def fit_lagrangian(self, X, y, num_epochs=5):
        """
        Train the SVM model using the given training data and labels.

        Parameters:
            X (array_like): Input data.
            y (array_like): Target values (-1 or 1 for binary classification).
            num_epochs (int, optional): Number of epochs for training. Defaults to 300.
            batch_size (int, optional): Batch size for mini-batch optimization. Defaults to 1024.
            optimizer (str, optional): Optimization algorithm ('SGD' or 'BGD'). Defaults to 'SGD'.
        """
        num_samples, input_dim = X.shape
        for epoch in range(num_epochs):
            for i in range(num_samples):
                gradient = self.compute_gradient(X, y, i)
                self.update_parameters(gradient)

    def compute_gradient(self, X, y, i):
        """
        Compute the gradient of the Lagrangian with respect to parameters.

        Parameters:
        -----------
        X : numpy.ndarray
            Input features (training data).
        y : numpy.ndarray
            Target labels.
        i : int
            Index of the current sample.

        Returns:
        -----------
        dict:
            Gradient of the Lagrangian with respect to parameters.
        """
        gradient = {}
        for j in range(len(self.alpha)):
            gradient[self.alpha[j]] = y[i] * y[j] * np.dot(X[i], X[j]) - 1 if j != i else 0
        gradient[self.b] = y[i]
        return gradient

    def update_parameters(self, gradient):
        """
        Update parameters using gradient descent.

        Parameters:
        -----------
        gradient : dict
            Gradient of the Lagrangian with respect to parameters.
        learning_rate : float
            Learning rate for gradient descent.
        """
        for parameter, grad in gradient.items():
            parameter.data -= self.learning_rate * grad

    def __repr__(self):
        """
        String representation of the SVM model.

        Returns:
        -----------
        str:
            String representation of the SVM model.
        """
        return f"SVM(alpha={self.alpha}, b={self.b})"


# Example usage:

if __name__ == '__main__':
    svm = SVM(input_dim=2)
    # print(svm)
    X_train = np.array([[5, 2], [23, 3], [3, 4]])
    y_train = np.array([1, -1, 1])
    svm.fit_lagrangian(X_train, y_train, num_epochs=10)
    y_pred = svm.predict(X_train)
    # print(svm)
    print(svm.alpha)
