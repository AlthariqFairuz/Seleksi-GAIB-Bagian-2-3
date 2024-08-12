import numpy as np
import pandas as pd

class GaussianNaiveBayes:
    """
    Create an instance of the Gaussian Naive Bayes algorithm
    """

    def fit(self, X, y):
        """
        Fit the training data to the model, calculate the mean, variance, and prior for each class
        """
        # Check for all unique values (classes) in the label
        self.classes = np.unique(y)

        # Store the mean, variance, and prior for each class
        self.mean = {}
        self.var = {}
        self.priors = {}

        # Calculate the mean, variance, and prior for each class
        for cls in self.classes:
            X_c = X[y == cls]
            print(X)
            self.mean[cls] = X_c.mean(axis=0)
            self.var[cls] = X_c.var(axis=0)
            self.priors[cls] = X_c.shape[0] / X.shape[0]

    # Calculate the probability of a feature vector x in a given class
    def _pdf(self, cls, x):
        """
        Calculate the probability of a feature vector x in a given class
        """
        mean = self.mean[cls]
        var = self.var[cls]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def predict_instance(self, x):
        """
        Predict the class label for a single sample
        """
        posteriors = []

        for cls in self.classes:
            prior = np.log(self.priors[cls])
            posterior = np.sum(np.log(self._pdf(cls, x)))
            posterior = prior + posterior
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        """
        Predict the class label for all of the samples
        """
        y_pred = [self.predict_instance(x) for x in X]
        return np.array(y_pred)