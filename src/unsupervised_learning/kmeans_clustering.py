import numpy as np

class KMeans:
    """
    Create an instance of the K-Means clustering algorithm
    """
    def __init__(self, n_clusters=3, max_iter=300, init='kmeans++'):
        """
        Constructor for the K-Means clustering algorithm
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.init = init

    def initialize_centroids(self, X):
        """
        Initialize the centroids
        """
        if self.init == 'random':
            indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
            return X[indices]
        
        elif self.init == 'kmeans++':
            centroids = [X[np.random.choice(X.shape[0])]]
            for _ in range(1, self.n_clusters):
                dist_sq = np.min([np.sum((X - centroid) ** 2, axis=1) for centroid in centroids], axis=0)
                probs = dist_sq / np.sum(dist_sq)
                next_centroid = X[np.random.choice(X.shape[0], p=probs)]
                centroids.append(next_centroid)
            return np.array(centroids)
        
        else:
            raise ValueError("Unknown initialization method")

    def fit(self, X):
        """
        Fit the training data to the model
        """
        self.centroids = self.initialize_centroids(X)
        for _ in range(self.max_iter):
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            self.labels_ = np.argmin(distances, axis=1)
            new_centroids = np.array([X[self.labels_ == i].mean(axis=0) for i in range(self.n_clusters)])
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids
        return self.labels_

    def predict(self, X):
        """
        Predict a new 
        """
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)