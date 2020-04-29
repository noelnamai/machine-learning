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

        for _ in range(self.iterations):
            for i in range(m):
                condition = y * (np.dot(x, self.w) + self.b)
                
                if condition >= 1:
                    self.w = self.w - self.alpha * (2 * self.gamma * self.w)
                else:
                    self.w = self.w - self.alpha * (2 * self.gamma * self.w + np.dot(x, y))
                    self.b = self.b - self.alpha * y

    def predict(self, x):
        y_pred = np.sign(np.dot(x, self.w) + self.b)

        return y_pred

    def linear_kernel(self, x, y):
        x = x.flatten()
        y = y.flatten()

        sim = np.dot(x, y.T)

        return sim

    def gaussian_kernel(self, x, y):
        x = x.flatten()
        y = y.flatten()

        sim = np.exp(- (np.sum((x - y) ** 2) / (2 * (self.gamma ** 2))))

        return sim
