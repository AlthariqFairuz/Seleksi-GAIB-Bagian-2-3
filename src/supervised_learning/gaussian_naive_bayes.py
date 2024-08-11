import numpy as np
import pandas as pd

class GaussianNaiveBayes:
    def fit(self, X, y):
        # Check for all unique values (classes) in the label
        self.classes = np.unique(y)

        # Store the mean, variance, and prior for each class
        self.mean = {}
        self.var = {}
        self.priors = {}

        # Calculate the mean, variance, and prior for each class
        for cls in self.classes:
            X_c = X[y == cls]
            self.mean[cls] = X_c.mean(axis=0)
            self.var[cls] = X_c.var(axis=0)
            self.priors[cls] = X_c.shape[0] / X.shape[0]

    # Calculate the probability of a feature vector x in a given class
    def _pdf(self, cls, x):
        mean = self.mean[cls]
        var = self.var[cls]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def predict_instance(self, x):
        posteriors = []

        for cls in self.classes:
            prior = np.log(self.priors[cls])
            posterior = np.sum(np.log(self._pdf(cls, x)))
            posterior = prior + posterior
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        y_pred = [self.predict_instance(x) for x in X]
        return np.array(y_pred)


# if __name__ == "__main__":
#     x_values = pd.DataFrame({
#         'height': [6, 5.92, 5.58, 5.92, 5, 5.5, 5.42, 5.75],
#         'weight': [180,190,170,165,100,150,130,150],
#         'foot': [12,11,12,10,6,8,7,9]
#     })

#     y = pd.Series([0,0,0,0,1,1,1,1])

#     gnb= GaussianNaiveBayes()
#     gnb.fit(x_values, y)
#     # gnb.predict(np.array([1.1, 1.1]))
#     print(gnb.predict(np.array([6, 13, 8])))
             