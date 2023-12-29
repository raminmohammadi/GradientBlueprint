import numpy as np

class Optimizer:
    @staticmethod
    def SGD(parameters, gradients, learning_rate, multiclass = False):
        if multiclass:
            for parameter in parameters:
                for p in parameter:
                    p.data += -learning_rate * np.mean(gradients[p.label])# + (regularization_term/len(parameters)) * p.data)
        else:
            for p in parameters:
                p.data += -learning_rate * np.mean(gradients[p.label])# + (regularization_term/len(parameters)) * p.data)

    @staticmethod
    def batch_gradient_descent(parameters, learning_rate, batch_size, multiclass = False):
        if multiclass:
            for parameter in parameters:
                for p in parameter:
                    p.data += -learning_rate * p.grad / batch_size # + (regularization_term/len(parameters)) * p.data)
        else:
            for p in parameters:
                p.data += -learning_rate * p.grad / batch_size # + (regularization_term/len(parameters)) * p.data)
                p.grad = 0.0