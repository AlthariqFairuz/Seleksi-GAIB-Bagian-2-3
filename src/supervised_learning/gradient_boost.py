import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from supervised_learning.decision_tree import DecisionTree

class GradientBoostingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, learning_rate=0.1, min_samples_split=4, max_depth=4, mode='gini'):
        """
        Create an instance of GradientBoostingClassifier.
        """
        self.n_estimators = n_estimators 
        self.learning_rate = learning_rate 
        self.min_samples_split = min_samples_split 
        self.max_depth = max_depth  
        self.mode = mode  

    def fit(self, X, Y):
        """
        Train the Gradient Boosting model by building multiple decision trees.
        """
        self.F0 = np.mean(Y)  # Initial prediction is from the mean of the label
        self.trees = [] # Save the trained trees

        # Gradual improvement
        residuals = Y - self.F0  # Calculate the residuals 
        for _ in range(self.n_estimators):
            tree = DecisionTree(min_samples_split=self.min_samples_split, max_depth=self.max_depth, mode=self.mode)
            tree.fit(X, residuals)  # Train the decision tree on the residuals
            predictions = tree.predict(X)  # Make predictions with the tree
            predictions = np.array(predictions)
            residuals -= self.learning_rate * predictions  # Update the residuals
            self.trees.append(tree)

    def predict(self, X):
        """
        Make predictions by summing the predictions from each tree.
        """
        pred = np.full(X.shape[0], self.F0)  # Initial prediction with the mean
        for tree in self.trees: 
            pred += self.learning_rate * np.array(tree.predict(X))  # Add the prediction from each tree
        return np.round(pred).astype(int)


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    data = load_iris()
    X = data.data
    Y = data.target
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, min_samples_split=4, max_depth=4, mode='gini')
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(Y_test, Y_pred))