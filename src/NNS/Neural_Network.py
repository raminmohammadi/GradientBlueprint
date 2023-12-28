import sys
import sys

# Get the parent directory
parent_directory = sys.path[0]  # Assumes the script is in the parent directory

# Add the parent directory to the Python path
sys.path.append(parent_directory)
from  src.Helpers.DrawGraph import draw_dot
from src.Gradient.Gradient import Variable
from Activations import Activations

class Neuron:
    def __init__(self, input_d, layer_index, node_index, activation = 'linear'):
        """
        @input_d: int, number of inputs to a given neuron
        """
        self.w = [Variable(random.uniform(-1, 1), label=f"Layer{layer_index}_node{node_index}_W{_}") for _ in range(input_d)]
        self.b = Variable(random.uniform(-1, 1), label=f"Layer{layer_index}_node{node_index}_b")
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
    def __init__(self, neuron_dim, layer_dim, layer_index, activation = 'linear'):
        self.neurons = [Neuron(neuron_dim, activation=activation, layer_index=layer_index, node_index=_) for _ in range(layer_dim)]
        
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
        self.layers = {"layer_{}".format(i+1):Layer(sizes[i], sizes[i+1], layer_index=i+1, activation=activation) for i in range(len(layers_dim))}
        
    def __call__(self,x):
        for layer in self.layers:
            x = self.layers[layer](x) # Forward pass
        return x

    def parameters(self):
        return [parameter for key, layer in self.layers.items() for parameter in layer.parameters()]


    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for key, layer in self.layers.items())}]"

    def __draw__(self, x):
        return draw_dot(self.__call__(x))
