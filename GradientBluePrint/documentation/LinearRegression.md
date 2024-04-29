# Linear Regression Class

Welcome to the documentation for the Linear Regression class! In this explanation, we'll dive into how the `Variable` class is utilized and how the concept of gradients (`grad`) is implemented within the code.

## Utilizing the `Variable` Class

The `Variable` class plays a crucial role in representing model parameters and performing computations during both the forward and backward passes of the linear regression model. Here's how it's utilized:

1. **Initialization of Model Parameters:**
   - In the `__init__` method of the `LinearRegression` class, the weights (`self.w`) and bias (`self.b`) are initialized as instances of the `Variable` class.
   - Each weight (`w`) is initialized with a random value between -1 and 1, and each has a label `W_i`, where `i` represents its index.
   - The bias (`b`) is similarly initialized with a random value between -1 and 1 and labeled as "b".

2. **Forward Pass (Prediction):**
   - In the `__call__` method of the `LinearRegression` class, the forward pass of the model is computed.
   - It calculates the predicted output (`y_hat`) by taking the dot product of the input features (`x`) and the weights (`w`), and then adding the bias (`b`).
   - The dot product is computed using list comprehension and the `sum()` function, where each weight (`w_i`) is multiplied by the corresponding input feature (`x_i`).

3. **Backward Pass (Gradient Calculation):**
   - The `backward()` method of the `Variable` class is called to perform the backward pass and calculate gradients.
   - Gradients represent the partial derivatives of the loss function with respect to each model parameter.
   - During the backward pass, gradients are accumulated in the `grad` attribute of each `Variable` object, representing how much the loss would change with a small change in the parameter value.

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
   - The `Optimizers.SGD()` and `Optimizers.batch_gradient_descent()` methods update the model parameters (`self.w` and `self.b`) based on the computed gradients and the specified learning rate.

In summary, the `Variable` class is essential for representing model parameters and performing computations with gradients during both the forward and backward passes of the linear regression model. Gradients are crucial for optimizing the model parameters to minimize the loss function and improve the model's predictive performance.
