import numpy as np

class Node:
    """
    Create a node for the Decision Tree
    """
    
    def __init__(self, feature_index= None, threshold= None, left= None, right= None, info_gain= None, value= None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    """
    Create an instance of the Decision Tree algorithm
    """
    def __init__(self, min_samples_split= 4, max_depth= 4):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def split(self, dataset, feature_index, threshold): 
        """
        Split the dataset into two parts based on the feature and threshold
        """
        dataset_left= np.array([row for row in dataset if row[feature_index] <= threshold])
        dataset_right= np.array([row for row in dataset if row[feature_index] > threshold])
        return dataset_left, dataset_right
    
    def entropy(self, y):
        """
        Calculate the entropy of the dataset
        """
        class_labels= np.unique(y)
        entropy= 0
        for cls in class_labels:
            p= len(y[y== cls]) / len(y)
            entropy += -p * np.log2(p)

        return entropy
    
    def gini_index(self, y): 
        """
        Calculate the Gini index of a tree node
        """   
        class_labels= np.unique(y)
        gini= 0
        for cls in class_labels:
            p= len(y[y== cls]) / len(y)
            gini += p * (1 - p)

        return gini
    
    def calculate_leaf_value(self, y): 
        """
        Calculate the leaf node value
        """ 
        Y= list(y)
        return max(Y, key= Y.count)
    
    def information_gain(self, parent_node, left_child, right_child, mode= "gini"):

        left_Weight= len(left_child) / len(parent_node)
        right_Weight= len(right_child) / len(parent_node)

        if mode== "gini":
            gain= self.gini_index(parent_node) - (left_Weight * self.gini_index(left_child) + right_Weight * self.gini_index(right_child))
        else:
            gain= self.entropy(parent_node) - (left_Weight * self.entropy(left_child) + right_Weight * self.entropy(right_child))
        
        return gain

    def get_best_split(self, dataset, num_features):
        best_split= {}
        max_info_gain= -float('inf')

        for feature_index in range (num_features):
            feature_values= dataset[:, feature_index]
            possible_thresholds= np.unique(feature_values)

            for thresholds in possible_thresholds:
                dataset_left, dataset_right= self.split(dataset, feature_index, thresholds)

                if len(dataset_left) > 0 and len(dataset_right) > 0:
                   y, y_left, y_right= dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]

                   curren_info_gain= self.information_gain(y, y_left, y_right, "gini")

                   if curren_info_gain > max_info_gain:
                       best_split['feature_index']= feature_index
                       best_split['threshold']= thresholds
                       best_split['data_left']= dataset_left
                       best_split['data_right']= dataset_right
                       best_split['info_gain']= curren_info_gain
                       max_info_gain= curren_info_gain

        return best_split

    def construct_tree(self, dataset, current_depth= 0):
        X, Y= dataset[:,:-1], dataset[:,-1]
        num_samples, num_features= X.shape

        if num_samples >= self.min_samples_split and current_depth <= self.max_depth: 
            best_split= self.get_best_split(dataset, num_samples, num_features)

            if best_split['info_gain'] > 0:
                left_subtree= self.construct_tree(best_split['data_left'], current_depth+1)
                right_subtree= self.construct_tree(best_split['data_right'], current_depth+1)
                return Node(best_split['feature_index'], best_split['threshold'], left_subtree, right_subtree, best_split['info_gain'])
            
        # Calculate leaf node
        leaf_value= self.calculate_leaf_value(Y)

        return Node(value= leaf_value)

    def print(self, tree= None, indent= " "):   

        if tree is None:
            tree= self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print(f"{tree.feature_index}, {tree.threshold} {tree.info_gain}")
            print(f"{indent}left: ", end= "")
            self.print(tree.left, indent + indent)
            print(f"{indent}right: ", end= "")
            self.print(tree.right, indent + indent)

    def fit(self, X, Y):
        dataset= np.concatenate((X, Y), axis= 1)
        self.root= self.construct_tree(dataset)

    def _predict(self, x, tree):
        
        if tree.value is not None:
            return tree.value

        feature_val= x[tree.feature_index]

        if feature_val <= tree.threshold:
            return self._predict(x, tree.left)
        else:
            return self._predict(x, tree.right)

    def predict(self, X):
        return [self._predict(x) for x in X]