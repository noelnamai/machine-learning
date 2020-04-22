#!/usr/bin/env python3

import numpy as np


class KNNClassifier(object):
    """
    Implementation of the K-Nearest Neighbors Classifier        
    """
    def __init__(self):
        pass

    def fit(self, x, y):
        """
        Parameters
        --------------------------------------------------
            x: ndarray of shape (n_samples, n_features)
            y: ndarray of shape (n_samples, 1)
        """
        if (isinstance(x, np.ndarray) == False):
            raise Exception(f"x should be an ndarray of shape (n_samples, n_features).")
        if (isinstance(y, np.ndarray) == False):
            raise Exception(f"y should be an ndarray of shape (n_samples, 1).")
        if ((x.shape[0] == y.shape[0]) == False):
            raise Exception( f"both x: {x.shape} and y: {y.shape} should be of length n_samples.")
        
        self.x_train = x
        self.y_train = y

    def predict(self, x, k = 1):
        """
        Parameters
        --------------------------------------------------
            x: ndarray of shape (n_samples, n_features)
            k: int number of nearest neighbors

        Returns
        --------------------------------------------------
            y_pred: ndarray of shape (n_samples, 1)
        """
        m      = x.shape[0]
        y_pred = np.zeros(m)

        for i in range(m):
            neighbors = self.find_neighbors(x[i], k)
            n_classes = self.y_train[neighbors]
            y_pred[i] = np.bincount(n_classes.flatten()).argmax()

        return y_pred
        
    def find_neighbors(self, x, k = 1):
        """
        Parameters
        --------------------------------------------------
            x: ndarray of shape (n_samples, n_features)
            k: int number of nearest neighbors

        Returns
        --------------------------------------------------
            neighbors: array of shape (k, 1)
        """
        neighbors = list()
        distances = self.euclidean_distance(self.x_train, x)
        
        for i, j in enumerate(distances.argsort()):
            if i < k:
                neighbors.append(j)
        
        return neighbors

    def euclidean_distance(self, centroid, x):
        """
        Parameters
        --------------------------------------------------
            x        : ndarray of shape (n_samples, n_features)
            centroids: ndarray of shape (n_features, 1)

        Returns
        --------------------------------------------------
            distance: ndarray of shape (n_samples, k)
        """
        #np.sqrt(np.sum((x[i] - centroid)**2))
        distance = np.linalg.norm(x - centroid, axis = 1)
        
        return distance



class KNNRegression(KNNClassifier):
    """
    Implementation of the K-Nearest Neighbors Regression        
    """
    def __init__(self):
        KNNClassifier.__init__(
            self
        )

    def predict(self, x, k = 1):
        """
        Parameters
        --------------------------------------------------
            x: ndarray of shape (n_samples, n_features)
            k: int number of nearest neighbors

        Returns
        --------------------------------------------------
            y_pred: ndarray of shape (n_samples, 1)
        """
        m      = x.shape[0]
        y_pred = np.zeros(m)

        for i in range(m):
            neighbors = self.find_neighbors(x[i], k)
            n_classes = self.y_train[neighbors]
            y_pred[i] = np.mean(n_classes)

        return y_pred
