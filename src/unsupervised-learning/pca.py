import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
    

    def fit(self, X):
        X_centered = X - np.mean(X, axis=0)
        covariance_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        self.eigenvectors = eigenvectors[:, sorted_idx[:self.n_components]]
        self.explained_variance_ = eigenvalues[sorted_idx[:self.n_components]] / np.sum(eigenvalues)

    def transform(self, X):
        X_centered = X - np.mean(X, axis=0)
        return np.dot(X_centered, self.eigenvectors)

    