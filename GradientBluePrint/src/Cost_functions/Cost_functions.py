import sys, os
# Get the parent directory
parent_directory = sys.path[0]  # Assumes the script is in the parent directory

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(parent_directory)))

from src.Gradient.Gradient import Variable

class CostFunction:
    @staticmethod
    def log_loss(y_hat, y):
        if y[0] == 1:
            loss = y_hat.log()
        else:
            loss = (1 - y_hat).log()
        return -loss
    
    @staticmethod    
    def categorical_cross_entropy_loss(y_hat, y):
        loss = y_hat[y[0]].log()
        return -loss
    
    @staticmethod
    def sse(y_hat, y):
        loss = (y_hat - y)**2
        return loss
    
    @staticmethod
    def hinge_loss(y_hat, y_true):
        """
        Computes the hinge loss for SVM.

        Parameters:
            y_hat (Variable): Predicted output.
            y_true (Variable): True labels.

        Returns:
            Hinge loss.
        """
        loss =  (1 - y_hat) * y_true
        loss.data = loss.data if loss.data >= 0 else 0
        return loss