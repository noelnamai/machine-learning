#!/usr/bin/env python3

import numpy as np


class Perceptron(object):
    """
    Fits a binary classification Linear Perceptron.

    Parameters
    --------------------------------------------------
        alpha     : float the learning rate
        iterations: int maximum number of iterations to be performed
    """
    def __init__(self, alpha = 0.01, iterations = 10000):
        self.b     = dict()
        self.w     = dict()
        self.alpha = alpha
        self.iters = iterations
        
    def fit(self, x, y):
        """
        Parameters
        --------------------------------------------------
            x: ndarray of shape (n_samples}, n_features)
            y: ndarray of shape (n_samples, 1)

        Returns
        --------------------------------------------------
            b: ndarray of shape (1, )
            w: ndarray of shape (n_features, 1)
        """
        if (isinstance(x, np.ndarray) == False):
            raise Exception(f"x should be an ndarray of shape (n_samples, n_features).")
        if (isinstance(y, np.ndarray) == False):
            raise Exception(f"y should be an ndarray of shape (n_samples, 1).")
        if ((x.shape[0] == y.shape[0]) == False):
            raise Exception( f"both x: {x.shape} and y: {y.shape} should be of length n_samples.")

        for i in np.unique(y):
            y_vs_all  = np.where(y == i, 1, 0)
            w, b      = self.gradient_descent(x, y_vs_all)
            self.b[i] = b
            self.w[i] = w

    def predict(self, x):
        """
        Parameters
        --------------------------------------------------
            x: ndarray of shape (n_samples, n_features)

        Returns
        --------------------------------------------------
            y_pred: ndarray of shape (n_samples, 1)
        """
        m      = x.shape[0]
        y_pred = np.zeros((m, 1))
        
        for i in range(m):
            stats = dict()
            for k, w in self.w.items():
                b        = self.b[k]
                stats[k] = self.perceptron(x[i], w, b)

            y_pred[i] = max(stats, key = lambda k: stats[k])

        return y_pred

    def perceptron(self, x, w, b):
        """
        Parameters
        --------------------------------------------------
            x: ndarray of shape (n_samples, n_features)
            b: ndarray of shape (1, )
            w: ndarray of shape (n_features, 1)

        Returns
        --------------------------------------------------
            y_pred: ndarray of shape (n_samples, 1) 
        """
        y = np.dot(x, w) + b

        return np.where(y >= 0, 1, 0)

    def gradient_descent(self, x, y):
        """
        Parameters
        --------------------------------------------------
            x: ndarray of shape (n_samples, 1 + n_features)
            y: ndarray of shape (n_samples, 1)

        Returns
        --------------------------------------------------
            b: ndarray of shape (1, )
            w: ndarray of shape (n_features, 1)
        """
        m = x.shape[0]
        n = x.shape[1]
        b = 0
        w = np.zeros(n)

        for _ in range(self.iters):
            for i in range(m):
                y_pred = self.perceptron(x[i], w, b)
                b      = b + self.alpha * (y[i] - y_pred)
                w      = w + self.alpha * (y[i] - y_pred) * x[i]

        return w, b
    