# Support Vector Machine (SVM) Class

Welcome to the documentation for the Support Vector Machine (SVM) classifier! In this explanation, we'll explore how the `Variable` class is utilized and the concept of gradients (`grad`) within the SVM class.

## Utilizing the `Variable` Class

The `Variable` class plays a crucial role in representing model parameters and facilitating computations within the SVM class. Let's delve into how it's utilized:

1. **Initialization of Model Parameters:**
   - In the `__init__` method of the `SVM` class, the weights (`self.w`), bias (`self.b`), and Lagrange multipliers (`self.alpha`) are initialized as instances of the `Variable` class.
   - Each weight (`w`) is initialized with a random value between -1 and 1, and each has a label `W_i`, where `i` represents its index.
   - The bias (`b`) is similarly initialized with a random value between -1 and 1 and labeled as "b".
   - The Lagrange multipliers (`alpha`) are initialized as instances of the `Variable` class with an initial value of 0.

2. **Forward Pass (Prediction):**
   - In the `__call__` method of the `SVM` class, the forward pass of the model is computed.
   - The dot product is computed between the input features (`x`) and the weights (`w`), and then the bias (`b`) is added.

3. **Backward Pass (Gradient Calculation):**
   - The `backward()` method of the `Variable` class is called to perform the backward pass and calculate gradients.
   - Gradients represent the partial derivatives of the hinge loss function with respect to each model parameter.
   - During the backward pass, gradients are accumulated in the `grad` attribute of each `Variable` object, indicating how much the loss would change with a small change in the parameter value.

## Concept of Gradients (`grad`)

1. **Gradient Accumulation:**
   - After computing the hinge loss function, gradients are computed with respect to each model parameter.
   - Gradients are accumulated in the `grad` attribute of each `Variable` object within the `backward()` method.
   - During the backward pass, gradients are accumulated iteratively for each parameter.

2. **Updating Model Parameters:**
   - Once gradients are computed, they are used to update the model parameters using gradient descent.
   - The `update_parameters()` method updates the model parameters based on the computed gradients and the specified learning rate.

## Training Methods

1. **Training using `fit()` Method:**
   - The `fit()` method trains the SVM model using the given training data and labels.
   - It performs mini-batch gradient descent to update model parameters iteratively over the specified number of epochs.
   - The `Optimizers.SGD()` or `Optimizers.batch_gradient_descent()` methods are used to update the parameters based on the computed gradients and the specified learning rate.

2. **Training using `fit_lagrangian()` Method:**
   - The `fit_lagrangian()` method trains the SVM model using the Lagrangian optimization approach.
   - It iteratively updates model parameters by computing gradients of the Lagrangian with respect to parameters for each sample in the training data.
   - Gradients are computed using the `compute_gradient()` method, and model parameters are updated using the `update_parameters()` method.

In summary, the `Variable` class is integral for representing model parameters and performing computations with gradients during both the forward and backward passes of the SVM model. Gradients play a crucial role in optimizing the model parameters to minimize the hinge loss function and enhance the model's predictive performance.

