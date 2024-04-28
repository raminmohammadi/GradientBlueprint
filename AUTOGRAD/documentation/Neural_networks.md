## Explanation of Neural Network Architecture

### Importing Libraries
- `sys`: Provides access to some variables used or maintained by the Python interpreter.
- `random`: Allows generating random numbers for initializing weights and biases.
- `List`: A type hint indicating that the variable should be a list.

### Setting Parent Directory
- Fetches the parent directory of the current script and adds it to the Python path, allowing importing modules from that directory.

### Neuron Class
- **Initialization**: 
  - `Neuron` instances are initialized with random weights (`self.w`) and a random bias (`self.b`).
  - Weights are represented as instances of the `Variable` class, each labeled with a unique identifier.
  - The `activation` parameter determines the activation function used by the neuron.
- **Forward Pass (`__call__`)**:
  - Computes the weighted sum of inputs and adds the bias.
  - Applies the activation function specified during initialization.
- **Parameters**:
  - Returns a list containing weights and bias.
- **Representation (`__repr__`)**:
  - Returns a string representation of the neuron indicating its activation function and the number of weights.

### Layer Class
- **Initialization**:
  - Initializes a layer with a specified number of neurons (`neuron_dim`) and activation function.
  - Creates a list of `Neuron` instances to form the layer.
- **Forward Pass (`__call__`)**:
  - Passes input through each neuron in the layer and returns the outputs.
- **Parameters**:
  - Returns a list containing parameters of all neurons in the layer.
- **Representation (`__repr__`)**:
  - Returns a string representation of the layer, showing the neurons it contains.

### MLP Class
- **Initialization**:
  - Initializes a Multi-Layer Perceptron (MLP) with specified input dimension, layer dimensions, and activation functions.
  - Creates layers based on the provided dimensions and activations.
- **Forward Pass (`__call__`)**:
  - Passes input through each layer sequentially, resulting in the output of the MLP.
- **Parameters**:
  - Returns a list containing all parameters of the MLP.
- **Representation (`__repr__`)**:
  - Returns a string representation of the MLP, showing its layers.
- **Drawing Computation Graph (`__draw__`)**:
  - Draws the computation graph of the MLP using the `draw_dot` function.

### Utilizing `Variable` Class
- The `Variable` class is used to represent weights (`self.w`) and biases (`self.b`) in neurons.
- It allows tracking gradients during backpropagation and updating parameters during optimization.
- Each weight is initialized with a random value and labeled uniquely within the neuron.
- Gradients are accumulated in the `grad` attribute of `Variable` objects during backpropagation.

### Node, Layer, and MLP
- A `Neuron` is a single node in a layer, responsible for processing input data.
- A `Layer` consists of multiple neurons (nodes) that collectively process input and generate output.
- An `MLP` is composed of multiple layers stacked sequentially, forming a deep neural network architecture.
