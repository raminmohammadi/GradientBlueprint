import sys
import math, random
import numpy as np

parent_directory = sys.path[0]
sys.path.append(parent_directory)

from src.Helpers.DrawGraph import draw_dot
from src.Gradient.Gradient import Variable
from src.Optimizers.optimizer import Optimizer
from src.NNS.Activation_Functions import Activations
from src.Cost_functions.Cost_functions import CostFunction


class LogisticRegression:
    def __init__(self, input_dim, multiclass=False, k = 2):
        if multiclass:
            self.w = [[Variable(random.uniform(-1, 1), label=f"Class{k}_W{_}") for _ in range(input_dim)] for k in range(k)]
            self.b = [Variable(random.uniform(-1, 1), label=f"Class{k}_b") for k in range(k)]
            self.activation = getattr(Activations, 'softmax')
            self.costFunction = getattr(CostFunction, 'categorical_cross_entropy_loss')
        else:
            self.w = [Variable(random.uniform(-1, 1), label=f"W{_}") for _ in range(input_dim)]
            self.b = Variable(random.uniform(-1, 1), label="b")
            self.activation = getattr(Activations, 'sigmoid')
            self.costFunction = getattr(CostFunction, 'log_loss')
        self.multiclass = multiclass
        
    def parameters(self):
        return self.w + [self.b]

    def predict(self, X):
        """
        Predicts the output for a given input x using the linear regression model.
        y_hat = intercept + X * w
        """
        ypreds = [self(x) for x in X]
        return ypreds
    
    def __call__(self, x):
        if self.multiclass:
            y_hat = [sum((w_i*x_i for w_i, x_i in zip(w, x.T)), b) for w, b in zip(self.w, self.b)]
            y_hat = self.activation(y_hat)
        else:
            y_hat = sum((w_i*x_i for w_i, x_i in zip(self.w, x.T)), self.b) # Forward pass
            y_hat = self.activation(y_hat)
        return y_hat
    
    def __draw__(self, x):
        return draw_dot(self.__call__(x))
     
    def regularizer(self, regularization_term):
        penalty_term = Variable(0, label='regularization_term')
        for param in self.w:
            penalty_term += (param ** 2)
        return penalty_term * regularization_term / (2 * len(self.w))
    
    
    def fit(self, X, y, learning_rate=0.001, num_epochs=300,
            batch_size=1024, optimizer='SGD', regularization_term = 0.05):
        
        for epoch in range(num_epochs):
            if batch_size is None:
                Xb, yb = X, y
            else:
                ri = np.random.permutation(X.shape[0])[:batch_size]
                Xb, yb = X[ri], y[ri]

            ypreds = self.predict(Xb)    

            # Forward pass
            if optimizer == 'SGD':
                if self.multiclass:
                    losses = [self.costFunction(ypred, y_) for ypred, y_ in zip(ypreds, yb)]   
                    gradients = {parameter.label: [] for parameters in self.parameters() for parameter in parameters}
                    
                    for k, loss in enumerate(losses):
                        loss.backward()
                        self.loss = loss
                        for parameters in self.parameters():
                            for parameter in parameters:
                                gradients[parameter.label].append(parameter.grad)
                                parameter.grad = 0.0

                else:
                    losses = [self.costFunction(ypred, y_) + self.regularizer(regularization_term) for ypred,
                          y_ in zip(ypreds, yb)]
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
                if self.multiclass:
                    loss = sum((self.costFunction(ypred, y_) for ypred, y_ in zip(ypreds, yb)),
                           Variable(0))               
                else:
                    loss = sum((self.costFunction(ypred, y_) for ypred, y_ in zip(ypreds, yb)),
                           Variable(0)) + self.regularizer(regularization_term) 
                loss.backward()

            # Update using specified optimizer
            if optimizer == 'SGD':
                Optimizer.SGD(self.parameters(), gradients, learning_rate, multiclass = self.multiclass)
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}, Loss: {sum([loss.data for loss in losses])}")

            elif optimizer == 'BGD':
                Optimizer.batch_gradient_descent(self.parameters(), learning_rate, batch_size, multiclass = self.multiclass)
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}, Loss: {loss.data}")
