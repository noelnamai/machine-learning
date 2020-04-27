#!/usr/bin/env python3

import numpy as np

from scipy.optimize import minimize


class CollaborativeFiltering(object):
    """
    Implementation of the Collaborative Filtering algorithm using 
    Low-Rank Matrix Factorization and Conjugate Gradient Descent.

    Parameters
    --------------------------------------------------
        alpha     : float the learning rate
        gamma     : float the regularization parameter
        iterations: int maximum number of iterations to be performed
        n_features: int the number of parameter or features to learn
    """
    def __init__(self, alpha = 0.01, gamma = 0.01, n_features = 10, iterations = 100):
        self.alpha      = alpha
        self.gamma      = gamma
        self.iters      = iterations
        self.n_features = n_features
        self.tolerance  = 1.0e-16

    def fit(self, y, is_rated):
        """
        Parameters
        --------------------------------------------------
            y       : ndarray of shape (n_items, n_users)
            is_rated: ndarray of shape (n_items, n_users)

        Returns
        --------------------------------------------------
            x         : ndarray of shape (n_items, n_features)
            theta     : ndarray of shape (n_users, n_features)
            opt_result: OptimizeResult the optimization result object
        """
        n_items = y.shape[0]
        n_users = y.shape[1]

        x      = np.random.random((n_items, self.n_features))
        theta  = np.random.random((n_users, self.n_features))
        params = np.concatenate((x.flatten(), theta.flatten()))        

        opt_result = minimize(
            fun = self.gradient, 
            x0 = params,
            args = (y, is_rated),
            method = "CG", 
            jac = True,
            tol = self.tolerance,
            options = {"maxiter": self.iters}
        )

        self.x     = np.reshape(opt_result.x[ :n_items * self.n_features], (n_items, self.n_features))
        self.theta = np.reshape(opt_result.x[n_items * self.n_features: ], (n_users, self.n_features))

        return opt_result

    def gradient(self, params, y, is_rated):
        """
        Parameters
        --------------------------------------------------
            params  :
            y       : ndarray of shape (n_items, n_users)
            is_rated: ndarray of shape (n_items, n_users)

        Returns
        --------------------------------------------------
            cost  : ndarray of shape (1, )
            params: ndarray of shape (n_features * (n_items + n_users), )
        """
        n_items = y.shape[0]
        n_users = y.shape[1]
        
        x     = np.reshape(params[ :n_items * self.n_features], (n_items, self.n_features))
        theta = np.reshape(params[n_items * self.n_features: ], (n_users, self.n_features))

        #np.dot(x, theta) = sum(theta.T * x)
        y_prime = np.dot(x, theta.T)

        cost = (1 / 2) * np.sum(((y_prime - y) * is_rated) ** 2)
        cost = cost + ((self.gamma / 2) * np.sum(x ** 2))
        cost = cost + ((self.gamma / 2) * np.sum(theta ** 2))

        x_grad     = self.alpha * ((np.dot(((y_prime - y) * is_rated), theta)) + self.gamma * x)
        theta_grad = self.alpha * ((np.dot(((y_prime - y) * is_rated).T, x)) + self.gamma * theta)

        params = np.concatenate((x_grad.flatten(), theta_grad.flatten()))

        return cost, params
