#!/usr/bin/env python3

import numpy as np


class KMeans(object):
    """
    Performs the K-Means Clustering algorithm i.e. Lloyd's algorithm a.k.a. Voronoi iteration or relaxation

    Parameters
    --------------------------------------------------
        k         : int the number of clusters to form i.e. the number of centroids to generate
        seed      : int seed used to initialize the pseudo-random number generator
        iterations: int maximum number of iterations to be performed
    """
    def __init__(self, k = 2, iterations = 100, seed = None):
        self.k         = k
        self.seed      = seed
        self.iters     = iterations
        self.cost      = list()
        self.classes   = None
        self.centroids = None

    def fit(self, x):
        """
        Parameters
        --------------------------------------------------
            x: ndarray of shape (n_samples, n_features)

        Returns
        --------------------------------------------------
            cost     : ndarray of shape (iterations, 1)
            centroids: ndarray of shape (n_features, 1)
            classes  : ndarray of shape (n_samples, 1)
        """
        if (isinstance(x, np.ndarray) == False):
            raise Exception(f"x should be an ndarray of shape (n_samples, n_features).")
        
        np.random.seed(seed = self.seed)

        m         = x.shape[0]
        classes   = np.zeros(m)
        distances = np.zeros([m, self.k])
        min_cost  = np.inf
        centroids = x[np.random.randint(m, size = self.k)]

        for _ in range(self.iters):
            for i in range(self.k):
                distances[:,i] = self.euclidean_distance(centroids[i], x)
            
            classes = np.argmin(distances, axis = 1)

            for i in range(self.k):
                centroids[i] = np.mean(x[classes == i], axis = 0)

            curr_cost = self.calculate_cost(x, classes, centroids)
            self.cost.append(curr_cost)

            if curr_cost < min_cost:
                min_cost       = curr_cost
                self.classes   = classes
                self.centroids = centroids

    def predict(self, x):
        """
        Parameters
        --------------------------------------------------
            x: ndarray of shape (n_samples, n_features)

        Returns
        --------------------------------------------------
            classes: ndarray of shape (n_samples, 1)
        """
        m         = x.shape[0]
        distances = np.zeros([m, self.k])
        
        for i in range(self.k):
            distances[:,i] = self.euclidean_distance(self.centroids[i], x)
        
        classes = np.argmin(distances, axis = 1)

        return classes

    def calculate_cost(self, x, classes, centroids):
        """
        Parameters
        --------------------------------------------------
            x        : ndarray of shape (n_samples, n_features)
            classes  : ndarray of shape (n_samples, 1)
            centroids: ndarray of shape (n_features, 1)

        Returns
        --------------------------------------------------
            cost: ndarray of shape (iterations, 1)
        """
        cltr_cost = np.zeros(self.k)

        for i in range(self.k):
            cltr_dist = self.euclidean_distance(centroids[i], x[classes == i])
            cltr_cost = np.append(cltr_cost, cltr_dist)

        cost = np.mean(cltr_cost)

        return cost

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



class KMedians(KMeans):
    """
    Performs the K-Medians Clustering algorithm.

    Parameters
    --------------------------------------------------
        k         : int the number of clusters to form i.e. the number of centroids to generate
        seed      : int seed used to initialize the pseudo-random number generator
        iterations: int maximum number of iterations to be performed
    """    
    def __init__(self, k = 2, iterations = 100, seed = None):
        KMeans.__init__(
            self, 
            k = k, 
            iterations = iterations, 
            seed = seed
        )

    def fit(self, x):
        """
        Parameters
        --------------------------------------------------
            x: ndarray of shape (n_samples, n_features)

        Returns
        --------------------------------------------------
            cost     : ndarray of shape (iterations, 1)
            centroids: ndarray of shape (n_features, 1)
            classes  : ndarray of shape (n_samples, 1)
        """
        np.random.seed(seed = self.seed)

        m         = x.shape[0]
        classes   = np.zeros(m)
        distances = np.zeros([m, self.k])
        min_cost  = np.inf
        centroids = x[np.random.randint(m, size = self.k)]

        for _ in range(self.iters):
            for i in range(self.k):
                distances[:,i] = self.euclidean_distance(centroids[i], x)
            
            classes = np.argmin(distances, axis = 1)

            for i in range(self.k):
                centroids[i] = np.median(x[classes == i], axis = 0)

            curr_cost = self.calculate_cost(x, classes, centroids)
            self.cost.append(curr_cost)

            if curr_cost < min_cost:
                min_cost       = curr_cost
                self.classes   = classes
                self.centroids = centroids
