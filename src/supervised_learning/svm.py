import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class SVM(BaseEstimator, ClassifierMixin):
    """ 
    Create an instance of the SVM model with soft margin 
    """

    def __init__(self, learning_rate=0.001, lambda_param=0.01, C=1.0, iterations=1000):
        """
        Constructor for the SVM model
        """
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param  # Regularization strength for L2 norm
        self.C = C # Regularization strength for slack variables (soft margin). Large C means less tolerance for misclassification
        self.iterations = iterations
        self.w = None
        self.b = None

    def update_weights(self):
        """
        Update the weights and bias
        """
        y_label = np.where(self.Y <= 0, -1, 1)

        for i, x_i in enumerate(self.X):
            condition = y_label[i] * (np.dot(x_i, self.w) - self.b) >= 1
            if condition:
                # No penalty for this point
                self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
            else:
                # Penalize this point by including the slack variable
                self.w -= self.learning_rate * (2 * self.lambda_param * self.w - self.C * np.dot(x_i, y_label[i]))
                self.b -= self.learning_rate * self.C * y_label[i]

    def fit(self, X, Y):
        """
        Fit the training data to the model
        """
        self.m, self.n = X.shape

        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        for _ in range(self.iterations):
            self.update_weights()

    def predict(self, X):
        """
        Predict the class label for a single sample
        """
        y_pred = np.sign(np.dot(X, self.w) - self.b)
        return np.where(y_pred <= -1, 0, 1)
