# Logistic Regression Class

Welcome to the documentation for the Logistic Regression class! In this explanation, we'll delve into how the `Variable` class is utilized and how the concept of gradients (`grad`) is implemented within the code.

## Utilizing the `Variable` Class

The `Variable` class serves a critical role in representing model parameters and facilitating computations during both the forward and backward passes of the logistic regression model. Let's explore how it's utilized:

1. **Initialization of Model Parameters:**
   - In the `__init__` method of the `LogisticRegression` class, the weights (`self.w`) and bias (`self.b`) are initialized as instances of the `Variable` class.
   - If the problem is multiclass, weights and biases are initialized for each class (indexed by `k`).
   - Each weight (`w`) is initialized with a random value between -1 and 1, and each has a label `Classk_W_i` for multiclass or `W_i` for binary, where `i` represents its index.
   - The bias (`b`) is similarly initialized with a random value between -1 and 1 and labeled as "Classk_b" for multiclass or "b" for binary.

2. **Forward Pass (Prediction):**
   - In the `__call__` method of the `LogisticRegression` class, the forward pass of the model is computed.
   - If the problem is multiclass, the model computes predicted probabilities for each class using the softmax activation function (`Activations.softmax`).
   - If it's binary, the sigmoid activation function (`Activations.sigmoid`) is used to compute the predicted probability of the positive class.
   - The dot product is computed between the input features (`x`) and the weights (`w`) for each class (for multiclass) or the single set of weights (for binary), and then the bias (`b`) is added.

3. **Backward Pass (Gradient Calculation):**
   - The `backward()` method of the `Variable` class is called to perform the backward pass and calculate gradients.
   - Gradients represent the partial derivatives of the loss function with respect to each model parameter.
   - During the backward pass, gradients are accumulated in the `grad` attribute of each `Variable` object, indicating how much the loss would change with a small change in the parameter value.

## Concept of Gradients (`grad`)

1. **Gradient Accumulation:**
   - After computing the loss function, gradients are computed with respect to each model parameter.
   - Gradients are accumulated in the `grad` attribute of each `Variable` object within the `backward()` method.
   - During the backward pass, gradients are accumulated iteratively for each parameter.

2. **Clearing Gradients:**
   - After each iteration (epoch) of training, gradients need to be cleared to avoid accumulation from previous iterations.
   - Gradients are cleared by resetting the `grad` attribute of each `Variable` object to zero (`parameter.grad = 0.0`) after updating the model parameters.

3. **Updating Model Parameters:**
   - Once gradients are computed, they are used to update the model parameters using optimization algorithms such as stochastic gradient descent (SGD) or batch gradient descent.
   - The `Optimizers.SGD()` and `Optimizers.batch_gradient_descent()` methods update the model parameters based on the computed gradients and the specified learning rate.

In summary, the `Variable` class is integral for representing model parameters and performing computations with gradients during both the forward and backward passes of the logistic regression model. Gradients play a crucial role in optimizing the model parameters to minimize the loss function and enhance the model's predictive performance.

