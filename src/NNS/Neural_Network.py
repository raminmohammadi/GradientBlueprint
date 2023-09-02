from Gradient import Variable
from Activation_Functions import Activations
import random


class Neuron:
    def __init__(self, input_d, activation = 'linear'):
        """
        @input_d: int, number of inputs to a given neuron
        """
        self.w = [Variable(random.uniform(-1, 1)) for _ in range(input_d)]
        self.b = Variable(random.uniform(-1, 1))
        self.activation = activation

    def __call__(self, x):
        """
        f(x) python will use __call__. 
        @x: list, input
        """
        act = sum((w_i*x_i for w_i, x_i in zip(self.w, x)), self.b)
        activation_function = getattr(Activations, self.activation)
        output = activation_function(act) if self.activation != 'linear' else act
        return output

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{self.activation} Neuron({len(self.w)})"

class Layer:
    def __init__(self, neuron_dim, layer_dim, activation = 'linear'):
        self.neurons = [Neuron(neuron_dim, activation) for _ in range(layer_dim)]
        
    def __call__(self, x):
        outputs = [neuron(x) for neuron in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self):
        params = [parameter for neuron in self.neurons for parameter in neuron.parameters()]
        return params

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP:
    def __init__(self, input_dim, layers_dim, activation = 'linear'):
        sizes = [input_dim] + layers_dim
        self.layers = [Layer(sizes[i], sizes[i+1], activation=activation) for i in range(len(layers_dim))]
        
    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [parameter for layer in self.layers for parameter in layer.parameters()]


    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
