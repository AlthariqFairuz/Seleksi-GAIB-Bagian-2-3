import numpy as np

class KNNeighbours:
    """
    Create an instance of the K-Nearest Neighbors algorithm
    """

    def __init__(self, k=3, metrics='euclidean'):    
        self.k = k
        self.metrics = metrics

    def fit(self, X, y):
        """
        Fit the training data to the model
        """
        self.X_train = X.values
        self.y_train = y.values

    def euclidean_distance(self, x1, x2):
        """
        Compute the Euclidean distance between two points
        """
        return np.sqrt(np.sum((x1 - x2)**2))
    
    def manhattan_distance(self, x1, x2):
        """
        Compute the Manhattan distance between two points
        """
        return np.sum(np.abs(x1 - x2))
    
    def minkowski_distance(self, x1, x2, p):
        """
        Compute the Minkowski distance between two points
        """
        return np.sum(np.abs(x1 - x2)**p)**(1/p)

    def _predict(self, x, p=2):
        """
        Predict the class label for a single sample
        """

        # Compute distances between x and all examples in the training set
        if self.metrics == 'euclidean':
            distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        elif self.metrics == 'manhattan':
            distances = [self.manhattan_distance(x, x_train) for x_train in self.X_train]
        elif self.metrics == 'minkowski':
            distances = [self.minkowski_distance(x, x_train, p) for x_train in self.X_train]
        else:
            raise ValueError('Invalid metrics')

        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k] # Return the indices of the k nearest neighbors from the lowest distance

        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Return the most common class label
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common
    
    def predict(self, X, p=2):
        """
        Predict the class labels for a set of samples
        """
        y_pred = [self._predict(x, p) for x in X.values]
        return np.array(y_pred)