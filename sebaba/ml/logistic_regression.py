#!/usr/bin/env python3

import numpy as np

##########################################################################################

class LogisticClassifier(object):
    """
    Performs Logistic Regression using Maximum Likelihood Estimation (MLE).

    Parameters
    --------------------------------------------------
        alpha     : float the learning rate
        normalize : bool
        iterations: int maximum number of iterations to be performed
    """
    def __init__(self, alpha = 0.01, iterations = 100000, normalize = True):
        self.theta      = dict()
        self.cost       = dict()
        self.alpha      = alpha
        self.normalize  = normalize
        self.iterations = iterations
        self.tolerance  = 1.0e-16

    def fit(self, x, y):
        """
        Parameters
        --------------------------------------------------
            x: ndarray of shape (n_samples}, n_features)
            y: ndarray of shape (n_samples, 1)

        Returns
        --------------------------------------------------
            cost : ndarray of shape (iterations, 1)
            theta: ndarray of shape (1 + n_features, 1)
        """
        if (isinstance(x, np.ndarray) == False):
            raise Exception(f"x should be an ndarray of shape (n_samples, n_features).")
        if (isinstance(y, np.ndarray) == False):
            raise Exception(f"y should be an ndarray of shape (n_samples, 1).")
        if ((x.shape[0] == y.shape[0]) == False):
            raise Exception( f"both x: {x.shape} and y: {y.shape} should be of length n_samples.")

        self.mu    = np.mean(x, axis = 0)
        self.sigma = np.std(x, axis = 0)
        
        x_scaled   = self.scale_and_normalize(x)

        for i in np.unique(y):
            y_one_vs_all  = np.where(y == i, 1, 0)
            cost, theta   = self.gradient_descent(x_scaled, y_one_vs_all)
            self.cost[i]  = cost
            self.theta[i] = theta

    def predict(self, x, threshold = 0.5):
        """
        Parameters
        --------------------------------------------------
            x: ndarray of shape (n_samples, n_features)

        Returns
        --------------------------------------------------
            y_pred: ndarray of shape (n_samples, 1)
        """
        m = x.shape[0]
        x = self.scale_and_normalize(x)
        y_pred = np.zeros((m, 1))

        for i in range(m):
            stats = dict()
            for key, theta in self.theta.items():
                stats[key] = self.sigmoid(np.dot(x[i], theta))
            y_pred[i] = max(stats, key = lambda k: stats[k])

        return y_pred

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

    def sigmoid(self, z):
        """
        Parameters
        --------------------------------------------------
            z: ndarray of shape (n_samples, 1 + n_features)

        Returns
        --------------------------------------------------
            probs: ndarray of shape (n_samples, 1) 
        """
        probs = 1 / (1 + np.exp(-z))
        
        return probs

    def cost_function(self, y, y_prime, theta):
        """
        Parameters
        --------------------------------------------------
            y      : ndarray of shape (n_samples, 1)
            y_prime: ndarray of shape (n_samples, 1)

        Returns
        --------------------------------------------------
            j_theta: ndarray of shape (1 + n_features, 1)
        """
        m = y.shape[0]
        n = y.shape[1]

        j_theta = -(1 / m) * np.sum(y * np.log(y_prime) + (1 - y) * np.log(1 - y_prime))
        
        return j_theta

    def gradient_descent(self, x, y):
        """
        Parameters
        --------------------------------------------------
            x: ndarray of shape (n_samples, 1 + n_features)
            y: ndarray of shape (n_samples, 1)

        Returns
        --------------------------------------------------
            cost : ndarray of shape (iterations, )
            theta: ndarray of shape (1 + n_features, 1)
        """
        m = x.shape[0]
        n = x.shape[1]

        cost      = list()
        theta     = np.zeros((n, 1))
        prev_cost = np.inf

        for _ in range(int(self.iterations)):
            #np.dot(x, theta) = sum(theta.T * x)
            y_prime   = self.sigmoid(np.dot(x, theta))
            theta     = theta - (self.alpha * (1 / m) * np.dot(x.T, (y_prime - y)))
            curr_cost = self.cost_function(y, y_prime, theta)
            cost.append(curr_cost)

            if (abs(prev_cost - curr_cost) > self.tolerance):
                prev_cost = curr_cost
            else:
                break

        return cost, theta

##########################################################################################

class RidgeClassifier(LogisticClassifier):
    """
    Performs Logistic Regression using Maximum Likelihood Estimation (MLE) with (L2) Ridge Regularization.

    Parameters
    --------------------------------------------------
        alpha     : float the learning rate
        gamma     : float the regularization parameter
        normalize : bool
        iterations: int maximum number of iterations to be performed
    """
    def __init__(self, alpha = 0.01, gamma = 0.01, iterations = 100000, normalize = True):
        self.gamma = gamma

        LogisticClassifier.__init__(
            self, 
            alpha = alpha, 
            iterations = iterations,
            normalize = normalize
        )

    def cost_function(self, y, y_prime, theta):
        """
        Parameters
        --------------------------------------------------
            y      : ndarray of shape (n_samples, 1)
            y_prime: ndarray of shape (n_samples, 1)

        Returns
        --------------------------------------------------
            j_theta: ndarray of shape (1 + n_features, 1)
        """
        m = y.shape[0]
        n = y.shape[1]

        j_theta = -(1 / m) * np.sum(y * np.log(y_prime) + (1 - y) * np.log(1 - y_prime)) + ((self.gamma / (2 * m)) * np.sum(theta ** 2))
        
        return j_theta

    def gradient_descent(self, x, y):
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

        cost      = list()
        theta     = np.zeros((n, 1))
        prev_cost = np.inf

        for _ in range(int(self.iterations)):
            #np.dot(x, theta) = sum(theta.T * x)
            y_prime   = self.sigmoid(np.dot(x, theta))
            theta     = theta * (1 - self.alpha * (self.gamma / m)) - (self.alpha * (1 / m) * np.dot(x.T, (y_prime - y)))
            curr_cost = self.cost_function(y, y_prime, theta)
            cost.append(curr_cost)
            
            if (abs(prev_cost - curr_cost) > self.tolerance):
                prev_cost = curr_cost
            else:
                break

        return cost, theta

##########################################################################################
