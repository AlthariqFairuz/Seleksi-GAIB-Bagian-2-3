import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000, reg_term=None, lambda_=0.01, loss_function='squared_error'):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.reg_term = reg_term
        self.lambda_ = lambda_
        self.loss_function = loss_function
        self.losses = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def compute_loss(self, y, y_pred):
        if self.loss_function == "squared_error":
            return 0.5 * np.mean((y - y_pred) ** 2)
        elif self.loss_function == "cross_entropy":
            return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        
    def update_weights(self):
        # General form of y_pred in logistic regression: Z= wX + b
        y_pred = self.sigmoid(np.dot(self.X, self.w) + self.b)
        loss = self.compute_loss(self.y, y_pred)
        self.losses.append(loss)
        
        # Compute the gradients
        dw = (1 / self.m) * np.dot(self.X.T, (y_pred - self.y))
        db = (1 / self.m) * np.sum(y_pred - self.y)
        
        # Update the weights
        if self.reg_term == 'l2':
            dw += (self.lambda_ / self.m) * self.w
        elif self.reg_term == 'l1':
            dw += (self.lambda_ / self.m) * np.sign(self.w)

        else:
            pass

        self.w -= self.learning_rate * dw
        self.b -= self.learning_rate * db

    def fit (self, X, y):

        # Define the number of samples and features
        self.m, self.n = X.shape

        # Initialize the weights and bias
        self.w = np.zeros(self.n)
        self.b = 0

        self.X = X
        self.y = y

        # Gradient descent
        for _ in range(self.iterations):
            self.update_weights()

    def predict(self, X):
        # Compute the predicted probabilities
        y_pred = self.sigmoid(np.dot(X, self.w) + self.b)
        return np.where( y_pred >= 0.5, 1, 0)