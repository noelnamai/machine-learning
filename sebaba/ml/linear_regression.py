#!/usr/bin/env python3

import numpy as np


class LinearRegression(object):
    """
    Performs Linear Regression using Ordinary Least Squares

    Attributes
    --------------------------------------------------
        alpha     : float the learning rate
        normalize : bool
        iterations: int maximum number of iterations to be performed
        tolerance : float
    """
    def __init__(self, alpha = 0.01, iterations = 10000, normalize = True):
        self.alpha      = alpha
        self.normalize  = normalize
        self.iterations = iterations
        self.tolerance  = 1.0e-16

    def fit(self, x, y):
        """
        Parameters
        --------------------------------------------------
            x: ndarray of shape (n_samples, n_features)
            y: ndarray of shape (n_samples, 1)

        Returns
        --------------------------------------------------
            cost : ndarray of shape (iterations, 1)
            theta: ndarray of shape (1 + n_features, 1)
        """
        if (isinstance(x, np.ndarray) == False):
            raise Exception(f"x should be an ndarray of shape (n_samples, n_features)")
        if (isinstance(y, np.ndarray) == False):
            raise Exception(f"y should be an ndarray of shape (n_samples, 1)")
        if ((x.shape[0] == y.shape[0]) == False):
            raise Exception( f"both x: {x.shape} and y: {y.shape} should be of length n_samples")
        
        self.mu    = np.mean(x, axis = 0)
        self.sigma = np.std(x, axis = 0)

        x_scaled    = self.scale_and_normalize(x)
        cost, theta = self.compute_gradient(x_scaled, y)

        self.cost  = cost
        self.theta = theta

    def predict(self, x_prime):
        """
        Parameters
        --------------------------------------------------
            x_prime: ndarray of shape (n_samples, n_features)

        Returns
        --------------------------------------------------
            y_prime: ndarray of shape (n_samples, 1)
        """
        x_prime = self.scale_and_normalize(x_prime)
        y_prime = np.dot(x_prime, self.theta)

        return y_prime

    def scale_and_normalize(self, x):
        """
        Parameters
        --------------------------------------------------
            x: ndarray of shape (n_samples, n_features)

        Returns
        --------------------------------------------------
            x: ndarray of shape (n_samples, 1 + n_features)
        """
        if self.normalize:
            x = (x - self.mu) / self.sigma
        
        x = np.insert(x, 0, 1, axis = 1)

        return x

    def compute_cost(self, y, y_prime, theta):
        """
        Parameters
        --------------------------------------------------
            y      : ndarray of shape (n_samples, 1)
            y_prime: ndarray of shape (n_samples, 1)
            theta  : ndarray of shape (1 + n_features, 1)

        Returns
        --------------------------------------------------
            cost: ndarray of shape (1 + n_features, 1)
        """
        m = y.shape[0]
        n = y.shape[1]

        cost = (1 / (2 * m)) * np.sum(np.square(y_prime - y))

        return cost

    def compute_gradient(self, x, y):
        """
        Parameters
        --------------------------------------------------
            x: ndarray of shape (n_samples, 1 + n_features)
            y: ndarray of shape (n_samples, 1)

        Returns
        --------------------------------------------------
            cost : ndarray of shape (iterations, 1)
            theta: ndarray of shape (1 + n_features, 1)
        """
        m = x.shape[0]
        n = x.shape[1]

        cost     = list()
        theta    = np.ones((n, 1))
        min_cost = np.inf
        
        for _ in range(int(self.iterations)):
            #np.dot(x, theta) = sum(theta.T * x)
            y_prime   = np.dot(x, theta)
            theta     = theta - (self.alpha * (1 / m) * np.dot(x.T, (y_prime - y)))
            curr_cost = self.compute_cost(y, y_prime, theta)
            cost.append(curr_cost)

            if (abs(min_cost - curr_cost) > self.tolerance):
                min_cost = curr_cost
            else:
                break

        return cost, theta


class RidgeRegression(LinearRegression):
    """
    Performs Linear Regression using Ordinary Least Squares with (L2) Ridge Regularization

    Attributes
    --------------------------------------------------
        alpha     : float the learning rate
        gamma     : float the regularization parameter
        normalize : bool
        iterations: int maximum number of iterations to be performed
        tolerance : float
    """
    def __init__(self, alpha = 0.01, gamma = 0.01, iterations = 10000, normalize = True):
        self.gamma = gamma
           
        LinearRegression.__init__(
            self, 
            alpha = alpha, 
            iterations = iterations, 
            normalize = normalize
        )

    def compute_cost(self, y, y_prime, theta):
        """
        Parameters
        --------------------------------------------------
            y      : ndarray of shape (n_samples, 1)
            y_prime: ndarray of shape (n_samples, 1)
            theta  : ndarray of shape (1 + n_features, 1)

        Returns
        --------------------------------------------------
            cost: ndarray of shape (1 + n_features, 1)
        """
        m = y.shape[0]
        n = y.shape[1]

        cost = (1 / (2 * m)) * (np.sum(np.square(y_prime - y)) + self.gamma * np.sum(theta ** 2))

        return cost

    def compute_gradient(self, x, y):
        """
        Parameters
        --------------------------------------------------
            x: ndarray of shape (n_samples, 1 + n_features)
            y: ndarray of shape (n_samples, 1)

        Returns
        --------------------------------------------------
            cost : ndarray of shape (iterations, 1)
            theta: ndarray of shape (1 + n_features, 1)
        """
        m = x.shape[0]
        n = x.shape[1]

        cost     = list()
        theta    = np.ones((n, 1))
        min_cost = np.inf
        
        for _ in range(int(self.iterations)):
            #np.dot(x, theta) = sum(theta.T * x)
            y_prime   = np.dot(x, theta)
            theta     = theta * (1 - self.alpha * (self.gamma / m)) - (self.alpha * (1 / m) * np.dot(x.T, (y_prime - y)))
            curr_cost = self.compute_cost(y, y_prime, theta)
            cost.append(curr_cost)

            if (abs(min_cost - curr_cost) > self.tolerance):
                min_cost = curr_cost
            else:
                break

        return cost, theta
