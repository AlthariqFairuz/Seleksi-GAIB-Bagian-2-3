import numpy as np
import matplotlib.pyplot as plt

# points = {"blue": [[3, 4, 5], [0, 3, 7], [9, 3, 2], [1, 2, 3], [2, 2, 1]],
#           "red": [[4, 5, 6], [7, 4, 5], [0, 4, 6], [1, 6, 6], [3, 5, 4]]}

# new_point = [3, 3, 5]

class KNNeighbours:
    def __init__(self, k=3, metrics='euclidean'):    
        self.k = k
        self.metrics = metrics

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))
    
    def manhattan_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2))
    
    def minkowski_distance(self, x1, x2, p):
        return np.sum(np.abs(x1 - x2)**p)**(1/p)

    def _predict(self, x, p=2):
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
        # print(distances)
        k_indices = np.argsort(distances)[:self.k] # Return the indices of the k nearest neighbors from the lowest distance

        # print("#######################################################################################")
        # print(k_indices) 
        # print("#######################################################################################")

        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # print(k_nearest_labels)

        # Return the most common class label
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common
    
    def predict(self, X, p=2):
        y_pred = [self._predict(x, p) for x in X]
        return np.array(y_pred)
    

# if __name__ == "__main__":
#     x_blue = np.array(points["blue"])
#     x_red = np.array(points["red"])
#     X = np.array(x_blue.tolist() + x_red.tolist())
#     y = np.array([0] * len(x_blue) + [1] * len(x_red))

#     knn = KNNeighbours(k=3, metrics='euclidean')
#     knn.fit(X, y)
#     print(knn.predict([new_point]))


#     # Plot the data and the new point to see whether it is classified as blue or red
#     fig = plt.figure(figsize=(15, 15))
#     ax = fig.add_subplot(projection= '3d')
#     ax.grid(True, color="#323232" )
#     ax.figure.set_facecolor("black")
#     ax.tick_params(axis='x', colors='white')
#     ax.tick_params(axis='y', colors='white')

#     for i in range(len(x_blue)):
#         ax.scatter(x_blue[i][0], x_blue[i][1], x_blue[i][2] , color="blue")

#     for i in range(len(x_red)):
#         ax.scatter(x_red[i][0], x_red[i][1],x_red[i][2], color="red")

#     predict = knn.predict([new_point])
#     color = "blue" if new_point == 0 else "red"
#     ax.scatter(new_point[0], new_point[1], new_point[2], color=color, marker = "*", s=200, zorder= 100)

#     for point in points["blue"]:
#         ax.plot([point[0], new_point[0]], [point[1], new_point[1]], [point[2], new_point[2]], color="blue", linestyle="--", linewidth=0.5)

#     for point in points["red"]:
#         ax.plot([point[0], new_point[0]], [point[1], new_point[1]], [point[2], new_point[2]], color="red", linestyle="--", linewidth=0.5)

#     plt.show()
    
    