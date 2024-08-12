import numpy as np

class LogisticRegression:
    """
    Create an instance of the Logistic Regression model
    """

    def __init__(self, learning_rate=0.01, iterations=1000, reg_term=None, lambda_=0.01):
        """
        Constructor for the Logistic Regression model
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.reg_term = reg_term
        self.lambda_ = lambda_
        self.losses = []

    def sigmoid(self, z):
        """
        Compute the sigmoid function
        """
        return 1 / (1 + np.exp(-z))
    
    def compute_loss(self, y, y_pred):
        """
        Compute the loss function
        """
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        
    def update_weights(self):
        """
        Update the weights and bias
        """
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

        # Update the weights and bias
        self.w -= self.learning_rate * dw
        self.b -= self.learning_rate * db

    def fit (self, X, y):
        """
        Fit the training data to the model
        """

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
        """
        Predict the class label for all of the samples
        """
        # Compute the predicted probabilities
        y_pred = self.sigmoid(np.dot(X, self.w) + self.b)
        return np.where( y_pred >= 0.5, 1, 0)