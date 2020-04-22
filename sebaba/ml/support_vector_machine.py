#!/usr/bin/env python3

import numpy as np


class SVMClassifier(object):

    def __init__(self, alpha = 0.01, gamma = 0.01, iterations = 10000):
        self.b          = None
        self.w          = None
        self.alpha      = alpha
        self.gamma      = gamma
        self.iterations = iterations

    def fit(self, x, y):
        m = x.shape[0]
        n = x.shape[1]

        self.b = 0
        self.w = np.zeros(n)

        for i in np.unique(y):
            y_ = np.where(y == i, 1, -1)
            for _ in range(self.iterations):
                for i in range(m):
                    condition = y_[i] * (np.dot(x[i], self.w) + self.b)
                    
                    if condition >= 1:
                        self.w = self.w - self.alpha * (2 * self.gamma * self.w)
                    else:
                        self.w = self.w - self.alpha * (2 * self.gamma * self.w + np.dot(x[i], y_[i]))
                        self.b = self.b - self.alpha * y_[i]

    def predict(self, x):
        y_pred = np.sign(np.dot(x, self.w) + self.b)

        return y_pred