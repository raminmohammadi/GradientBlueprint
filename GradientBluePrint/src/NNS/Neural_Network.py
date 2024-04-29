import sys
import random
from typing import List

# Get the parent directory
parent_directory = sys.path[0]  # Assumes the script is in the parent directory

# Add the parent directory to the Python path
sys.path.append(parent_directory)
from src.Helpers.DrawGraph import draw_dot
from src.Gradient.Gradient import Variable
from Activation_Functions import Activations

class Neuron:
    def __init__(self, input_d, layer_index, node_index, activation='linear'):
        """
        Initializes a neuron with random weights and bias.

        Parameters:
            input_d (int): Number of inputs to the neuron.
            layer_index (int): Index of the layer the neuron belongs to.
            node_index (int): Index of the neuron within the layer.
            activation (str, optional): Activation function type. Defaults to 'linear'.
        """
        self.w = [Variable(random.uniform(-1, 1), label=f"Layer{layer_index}_node{node_index}_W{_}") for _ in range(input_d)]
        self.b = Variable(random.uniform(-1, 1), label=f"Layer{layer_index}_node{node_index}_b")
        self.activation = activation

    def __call__(self, x):
        """
        Computes the output of the neuron given input.

        Parameters:
            x (list): Input to the neuron.

        Returns:
            float: Output of the neuron.
        """
        act = sum((w_i * x_i for w_i, x_i in zip(self.w, x)), self.b)
        activation_function = getattr(Activations, self.activation)
        output = activation_function(act) if self.activation != 'linear' else act
        return output

    def parameters(self):
        """
        Get the parameters (weights and bias) of the neuron.

        Returns:
            list: List of parameters.
        """
        return self.w + [self.b]

    def __repr__(self):
        """
        Representation of the neuron.

        Returns:
            str: String representation of the neuron.
        """
        return f"{self.activation} Neuron({len(self.w)})"

class Layer:
    def __init__(self, neuron_dim, layer_dim, layer_index, activation='linear'):
        """
        Initializes a layer with neurons.

        Parameters:
            neuron_dim (int): Dimensionality of each neuron in the layer.
            layer_dim (int): Number of neurons in the layer.
            layer_index (int): Index of the layer.
            activation (str, optional): Activation function type. Defaults to 'linear'.
        """
        self.neurons = [Neuron(neuron_dim, activation=activation,
                               layer_index=layer_index, node_index=_) for _ in range(layer_dim)]
        
    def __call__(self, x):
        """
        Computes the output of the layer given input.

        Parameters:
            x (list): Input to the layer.

        Returns:
            list: Output of the layer.
        """
        outputs = [neuron(x) for neuron in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self):
        """
        Get the parameters of all neurons in the layer.

        Returns:
            list: List of parameters.
        """
        params = [parameter for neuron in self.neurons for parameter in neuron.parameters()]
        return params

    def __repr__(self):
        """
        Representation of the layer.

        Returns:
            str: String representation of the layer.
        """
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP:
    def __init__(self, input_dim: int, layers_dim: List[int], activations: List[str] = ['linear']):
        """
        Initializes a Multi-Layer Perceptron.

        Parameters:
            input_dim (int): Dimensionality of the input data.
            layers_dim (List[int]): List of integers representing dimensions of each layer.
            activations (List[str], optional): List of activation function types.
                Defaults to ['linear'].
        """
        sizes = [input_dim] + layers_dim
        
        if len(activations)==1:
            activations = activations * len(layers_dim)
        
        self.layers = {"layer_{}".format(i+1):Layer(sizes[i], sizes[i+1],
                                                    layer_index=i+1,
                                                    activation=activations[i]) for i in range(len(layers_dim))}
        
    def __call__(self,x):
        """
        Forward pass through the MLP.

        Parameters:
            x (list): Input data.

        Returns:
            list: Output of the MLP.
        """
        for layer in self.layers:
            x = self.layers[layer](x) # Forward pass
        return x

    def parameters(self):
        """
        Get all parameters of the MLP.

        Returns:
            list: List of parameters.
        """
        return [parameter for key, layer in self.layers.items() for parameter in layer.parameters()]


    def __repr__(self):
        """
        Representation of the MLP.

        Returns:
            str: String representation of the MLP.
        """
        return f"MLP of [{', '.join(str(layer) for key, layer in self.layers.items())}]"

    def __draw__(self, x):
        """
        Draw the computation graph of the MLP.

        Parameters:
            x (list): Input data.

        Returns:
            str: Graph in DOT format.
        """
        return draw_dot(self.__call__(x))
