from sklearn.metrics import pairwise_distances
import numpy as np

class DBSCAN:
    """
    Create an instance of the DBSCAN clustering algorithm
    """
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean'):
        """
        Constructor for the DBSCAN clustering
        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

    def fit(self, X):
        """
        Fit the training data to the model
        """
        distance_matrix = pairwise_distances(X, metric=self.metric)
        labels = np.full(X.shape[0], -1)
        cluster_id = 0

        for i in range(X.shape[0]):
            if labels[i] != -1:
                continue

            neighbors = np.where(distance_matrix[i] <= self.eps)[0]

            if len(neighbors) < self.min_samples:
                labels[i] = -1  # noise

            else:
                labels[i] = cluster_id
                queue = list(neighbors)
                while queue:
                    neighbor = queue.pop(0)
                    if labels[neighbor] == -1:
                        labels[neighbor] = cluster_id
                    if labels[neighbor] != -1:
                        continue
                    labels[neighbor] = cluster_id
                    new_neighbors = np.where(distance_matrix[neighbor] <= self.eps)[0]
                    if len(new_neighbors) >= self.min_samples:
                        queue.extend(new_neighbors)
                cluster_id += 1

        self.labels_ = labels
        return self.labels_
