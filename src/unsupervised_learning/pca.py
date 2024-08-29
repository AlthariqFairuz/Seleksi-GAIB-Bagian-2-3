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
        covariance_matrix = np.cov(X, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        self.eigenvectors = eigenvectors[:, sorted_idx[:self.n_components]]
        self.explained_variance_ = eigenvalues[sorted_idx[:self.n_components]] / np.sum(eigenvalues)
        return self

    def transform(self, X):
        """
        Transform the data
        """
        return np.dot(X, self.eigenvectors)
    
    def fit_transform(self, X):
        """
        Fit the training data to the model and transform it
        """
        self.fit(X)
        return self.transform(X)