#!/usr/bin/env python3

import numpy as np

from scipy.optimize import minimize


class CollaborativeFiltering(object):
    """
    Implementation of the Collaborative Filtering Learning algorithm.

    Parameters
    --------------------------------------------------
        alpha     : float the learning rate
        gamma     : float the regularization parameter
        iterations: int maximum number of iterations to be performed
        n_features: int the number of features to learn
    """
    def __init__(self, alpha = 0.01, gamma = 0.01, n_features = 10, iterations = 100):
        self.alpha      = alpha
        self.gamma      = gamma
        self.iters      = iterations
        self.n_features = n_features
        self.tolerance  = 1.0e-16

    def fit(self, y, is_rated):
        n_movies = y.shape[0]
        n_users  = y.shape[1]

        x     = np.random.random((n_movies, self.n_features))
        theta = np.random.random((n_users, self.n_features))

        params = np.concatenate((x.flatten(), theta.flatten()))        
        y_norm = self.normalize_ratings(y, is_rated)

        opt_result = minimize(
            fun = self.gradient, 
            x0 = params,
            args = (y_norm, is_rated),
            method = "CG", 
            jac = True,
            tol = self.tolerance,
            options = {"maxiter": self.iters}
        )

        self.x     = np.reshape(opt_result.x[:n_movies * self.n_features], (n_movies, self.n_features))
        self.theta = np.reshape(opt_result.x[n_movies * self.n_features:], (n_users, self.n_features))

        return opt_result

    def gradient(self, params, y, is_rated):
        n_movies = y.shape[0]
        n_users  = y.shape[1]
        
        x     = np.reshape(params[ :n_movies * self.n_features], (n_movies, self.n_features))
        theta = np.reshape(params[n_movies * self.n_features: ], (n_users, self.n_features))

        #np.dot(x, theta) = sum(theta.T * x)
        y_prime = np.dot(x, theta.T)

        cost = (1 / 2) * np.sum(((y_prime - y) * is_rated) ** 2)
        cost = cost + ((self.gamma / 2) * np.sum(x ** 2))
        cost = cost + ((self.gamma / 2) * np.sum(theta ** 2))

        x_grad     = self.alpha * ((np.dot(((y_prime - y) * is_rated), theta)) + self.gamma * x)
        theta_grad = self.alpha * ((np.dot(((y_prime - y) * is_rated).T, x)) + self.gamma * theta)

        params = np.concatenate((x_grad.flatten(), theta_grad.flatten()))

        return cost, params

    def normalize_ratings(self, y, is_rated):
        n_movies = y.shape[0]
        n_users  = y.shape[1]

        y_mean = np.zeros((n_movies, 1))
        y_norm = np.zeros((n_movies, n_users))

        for i in range(n_movies):
            idx            = is_rated[i, :] != 0
            y_mean[i]      = np.mean(y[i, idx])
            y_norm[i, idx] = y[i, idx] - y_mean[i]

        self.y_mean = y_mean

        return y_norm
