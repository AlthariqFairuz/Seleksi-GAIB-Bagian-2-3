import numpy as np

# Activation functions
class ActivationFunction:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def linear_derivative(x):
        return np.ones_like(x)

    @staticmethod
    def softmax(x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    @staticmethod
    def softmax_derivative(x):
        # For usage with cross-entropy, this typically simplifies during backpropagation. However, this is not the general derivative of softmax.
        return x * (1 - x)

# Loss functions
class LossFunction:
    @staticmethod
    def mean_squared_error(y, y_pred):
        return np.mean(np.power(y - y_pred, 2))

    @staticmethod
    def mean_squared_error_derivative(y, y_pred):
        return 2 * (y_pred - y) / y.size

    @staticmethod
    def binary_cross_entropy(y, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Prevent log(0)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    @staticmethod
    def binary_cross_entropy_derivative(y, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Prevent division by 0
        return (y_pred - y) / (y_pred * (1 - y_pred)) 

# Regularization
class Regularization:
    @staticmethod
    def l1(lambda_reg, weights):
        return lambda_reg * np.sign(weights)

    @staticmethod
    def l2(lambda_reg, weights):
        return lambda_reg * weights

    @staticmethod
    def none():
        return 0

# Fully Connected Layer
class FullyConnectedLayer:
    def __init__(self, input_size, output_size, activation='relu', lambda_reg=0.0, regularization='l2', init_method= None):
        self.activation = activation
        self.lambda_reg = lambda_reg
        self.regularization = regularization

        # Weight Initialization
        if init_method == 'xavier':
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(1 / input_size)
        elif init_method == 'he':
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        else:
            self.weights = np.random.randn(input_size, output_size) * 0.1

        self.biases = np.zeros((1, output_size))

    def activate(self, z):
        if self.activation == 'sigmoid':
            return ActivationFunction.sigmoid(z)
        elif self.activation == 'relu':
            return ActivationFunction.relu(z)
        elif self.activation == 'linear':
            return ActivationFunction.linear(z)
        elif self.activation == 'softmax':
            return ActivationFunction.softmax(z)

    def activate_derivative(self, a):
        if self.activation == 'sigmoid':
            return ActivationFunction.sigmoid_derivative(a)
        elif self.activation == 'relu':
            return ActivationFunction.relu_derivative(a)
        elif self.activation == 'linear':
            return ActivationFunction.linear_derivative(a)
        elif self.activation == 'softmax':
            return ActivationFunction.softmax_derivative(a)

    def forward(self, input_data):
        self.input_data = input_data
        self.z = np.dot(input_data, self.weights) + self.biases
        self.a = self.activate(self.z)
        return self.a

    def backward(self, output_error, learning_rate):
        activation_derivative = self.activate_derivative(self.a)
        self.delta = output_error * activation_derivative
        input_error = np.dot(self.delta, self.weights.T)

        # Regularization
        if self.regularization == 'l1':
            reg_term = Regularization.l1(self.lambda_reg, self.weights)
        elif self.regularization == 'l2':
            reg_term = Regularization.l2(self.lambda_reg, self.weights)
        else:
            reg_term = Regularization.none()

        # Update weights and biases
        self.weights -= learning_rate * (np.dot(self.input_data.T, self.delta) + reg_term)
        self.biases -= learning_rate * np.sum(self.delta, axis=0, keepdims=True)

        return input_error

# Neural Network
class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss_function = None
        self.loss_function_derivative = None

    def add_layer(self, layer):
        self.layers.append(layer)

    def set_loss_function(self, loss_function):
        if loss_function == 'mse':
            self.loss_function = LossFunction.mean_squared_error
            self.loss_function_derivative = LossFunction.mean_squared_error_derivative
        elif loss_function == 'binary_cross_entropy':
            self.loss_function = LossFunction.binary_cross_entropy
            self.loss_function_derivative = LossFunction.binary_cross_entropy_derivative
        else:
            raise ValueError('Invalid loss function')

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, y, y_pred, learning_rate):
        error = self.loss_function_derivative(y, y_pred)
        for layer in reversed(self.layers):
            error = layer.backward(error, learning_rate)

    def train(self, X, y, epochs, batch_size, learning_rate):
        y= np.reshape(y, (-1, 1))
        for epoch in range(epochs):
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]

                y_pred = self.forward(X_batch)
                self.backward(y_batch, y_pred, learning_rate)

            loss = self.loss_function(y, self.forward(X))
            print(f'Epoch {epoch + 1}/{epochs} - Loss: {loss}')

    def predict(self, X):
        return self.forward(X)