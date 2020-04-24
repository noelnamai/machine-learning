#!/usr/bin/env python3

import numpy as np


class CollaborativeFiltering(object):
    """
    Implementation of the Collaborative Filtering Learning algorithm.

    Parameters
    --------------------------------------------------
        iterations: int maximum number of iterations to be performed
    """
    def __init__(self, iterations = 100):
        self.iters     = iterations
        self.tolerance = 1.0e-16

    def fit(self, y, is_rated):
        m = y.shape[0]
        n = y.shape[1]

        theta  = np.ones((n, 100))
        x      = np.zeros((m, 100))        
        y_norm = self.normalize_ratings(y, is_rated)

        self.gradient_descent(x, y_norm, theta, is_rated)

    def cost_function(self, y, y_prime, is_rated):
        cost = (1 / 2) * np.sum(((y_prime - y) ** 2) * is_rated)

        return cost

    def gradient_descent(self, x, y, theta, is_rated):
        self.cost = list()

        prev_cost = np.inf

        for _ in range(int(self.iters)):
            #np.dot(x, theta) = sum(theta.T * x)
            y_prime   = np.dot(x, theta.T)
            x         = x - np.dot(((y_prime - y) * is_rated), theta)
            theta     = theta - np.dot(((y_prime - y) * is_rated).T, x)
            curr_cost = self.cost_function(y, y_prime, is_rated)
            self.cost.append(curr_cost)

        self.x     = x
        self.theta = theta

    def normalize_ratings(self, y, is_rated):
        m = y.shape[0]
        n = y.shape[1]

        y_mean = np.zeros((m, 1))
        y_norm = np.zeros((m, n))

        for i in range(m):
            idx            = is_rated[i,:] != 0
            y_mean[i]      = np.mean(y[i, idx])
            y_norm[i, idx] = y[i, idx] - y_mean[i]

        return y_norm
