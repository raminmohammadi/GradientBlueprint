from Gradient import Variable
import math

class Activations(Variable):
    def sigmoid(self):
        n = self.data
        t = 1/(1 + math.exp(-n))
        output = Variable(t, _children=(self, ), _op = 'sigmoid')

        def _backward():
            self.grad += t * (1 - t) * output.grad

        output._backward = _backward        
        return output

    def tanh(self):
        n = self.data
        t = (math.exp(2*n) - 1)/(math.exp(2*n) + 1)
        output = Variable(t, _children=(self, ), _op = 'tanh')

        def _backward():
            self.grad += (1 - t**2) * output.grad

        output._backward = _backward        
        return output

    def relu(self):
        """
        The product of gradients of ReLU function doesn't end up converging to 0 as the value is either 0 or 1.
        If the value is 1, the gradient is back propagated as it is. If it is 0, then no gradient is backpropagated from that point backwards.
        """
        n = self.data
        t = n if n > 0 else 0
        output = Variable(t, _children=(self, ), _op = 'relu')

        def _backward():
            if t > 0:
                self.grad += 1 * output.grad
            else:
                self.grad += 0 * output.grad

        output._backward = _backward        
        return output

    def leaky_relu(self):
        """
        """
        n = self.data
        t = n if n > 0 else 0.01*n
        output = Variable(t, _children=(self, ), _op = 'leaky relu')

        def _backward():
            if t > 0:
                self.grad += 1 * output.grad
            else:
                self.grad += 0.01 * output.grad

        output._backward = _backward        
        return output        

    # @staticmethod
    # def softmax(self, xs):
    #     datas = [x.data for x in xs]
    #     e_x = [np.exp(a - max(datas)) for a in [self.data] + datas]
    #     smx = e_x[0] / sum(e_x)
    #     output = Variable(smx, _children=(self, ), _op = 'softmax')

        # def _backward():
        #     if  > 0:
        #         self.grad += 1 * output.grad
        #     else:
        #         self.grad += 0.01 * output.grad
                
        # output._backward = _backward        
        # return output    