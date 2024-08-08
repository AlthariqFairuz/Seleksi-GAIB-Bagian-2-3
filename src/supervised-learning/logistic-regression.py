import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000, reg_term=None, lambda_=0.01, loss_function='squared_error'):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.reg_term = reg_term
        self.lambda_ = lambda_
        self.loss_function = loss_function
        self.theta = None
        self.losses = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def compute_loss(self, y, y_pred):
        if self.loss_function == "squared_error":
            return 0.5 * np.mean((y - y_pred) ** 2)
        elif self.loss_function == "cross_entropy":
            return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        
    def fit (self, X, y):
        m, n= X.shape
        self.theta = np.zeros(n)
        self.losses = []

        for i in range (self.iterations):
            z = np.dot(X, self.theta)