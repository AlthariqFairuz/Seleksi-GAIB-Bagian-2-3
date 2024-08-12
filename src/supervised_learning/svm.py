import numpy as np

class SVM:
    """ 
    Create an instance of the SVM model 
    """

    def __init__(self, learning_rate=0.001, lambda_param=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.iterations = iterations
        self.w = None
        self.b = None


    def update_weights(self,):
        """
        Update the weights and bias
        """
        y_label= np.where(self.Y <= 0, -1, 1)

        for i, x_i in enumerate(self.X):
            condition= y_label[i] * np.dot(x_i, self.w) - self.b >= 1
            if condition:
                self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
            else:
                self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y_label[i]))
                self.b -= self.learning_rate * y_label[i]


    def fit(self, X, Y):
        """
        Fit the training data to the model
        """

        self.m, self.n = X.shape

        self.w= np.zeros(self.n)
        self.b= 0
        self.X = X
        self.Y = Y

        for _ in range(self.iterations):
            self.update_weights()

    def predict(self):
        y_pred= np.where(np.sign(np.dot(self.X, self.w) - self.b) <= -1, 0, 1)
        # return y_pred()
        return y_pred