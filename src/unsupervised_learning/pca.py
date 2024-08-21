import numpy as np

class PCA:
    """
    Create an instance of the PCA model
    """
    def __init__(self, n_components):
        """
        Constructor for the PCA model
        """
        self.n_components = n_components

    def fit(self, X):
        """
        Fit the training data to the model
        """
        X_centered = X - np.mean(X, axis=0)
        covariance_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        self.eigenvectors = eigenvectors[:, sorted_idx[:self.n_components]]
        self.explained_variance_ = eigenvalues[sorted_idx[:self.n_components]] / np.sum(eigenvalues)
        return self

    def transform(self, X):
        X_centered = X - np.mean(X, axis=0)
        return np.dot(X_centered, self.eigenvectors)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)