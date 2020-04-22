#!/usr/bin/env python3

import numpy as np


class PCA(object):
    """
    Performs Principal Component Analysis

    Parameters
    --------------------------------------------------
        k: int the number of clusters to form i.e. the number of centroids to generate
    """    
    def __init__(self, k = 1):
        self.k = k
    
    def fit(self, x):
        """
        Parameters
        --------------------------------------------------
            x: ndarray of shape (n_samples, n_features)

        Returns
        --------------------------------------------------
            eigen_values : ndarray of shape (k, )
            eigen_vectors: ndarray of shape (n_features, k)
        """
        m = x.shape[0]
        n = x.shape[1]

        self.mu    = np.mean(x, axis = 0)
        self.sigma = np.std(x, axis = 0)

        if min(m, n) >= self.k:
            x_scaled = self.scale_and_normalize(x)     
            c_matrix = self.compute_covariance_matrix(x_scaled)
            w, v     = self.compute_eigen_vectors(c_matrix)

            self.eigen_values  = w[  :self.k]
            self.eigen_vectors = v[:,:self.k]
        else:
            raise ValueError(f"k = {self.k} must be between 0 and min(m, n) = {min(m, n)}")

    def transform(self, x):
        """
        Parameters
        --------------------------------------------------
            x: ndarray of shape (n_samples, n_features)

        Returns
        --------------------------------------------------
            x_trans: ndarray of shape (n_samples, n_features)
        """
        x_scaled = self.scale_and_normalize(x)
        x_trans  = np.dot(x_scaled, self.eigen_vectors)

        return x_trans

    def inverse_transform(self, x):
        """
        Parameters
        --------------------------------------------------
            x: ndarray of shape (n_samples, k)

        Returns
        --------------------------------------------------
            x_orig: ndarray of shape (n_samples, n_features)
        """
        x_inv  = np.dot(x, self.eigen_vectors.T)
        x_orig = (x_inv * self.sigma) + self.mu

        return x_orig
        
    def scale_and_normalize(self, x):
        """
        Parameters
        --------------------------------------------------
            x: ndarray of shape (n_samples, n_features)

        Returns
        --------------------------------------------------
            x: ndarray of shape (n_samples, n_features)
        """
        x = (x - self.mu) / self.sigma

        return x

    def compute_covariance_matrix(self, x):
        """
        Parameters
        --------------------------------------------------
            x: ndarray of shape (n_samples, n_features)

        Returns
        --------------------------------------------------
            matrix: ndarray of shape (n_features, n_features)
        """
        m      = x.shape[0]
        matrix = (1 / (m - 1)) * np.dot(x.T, x)

        return matrix

    def compute_eigen_vectors(self, m):
        """
        Parameters
        --------------------------------------------------
            m: ndarray of shape (n_features, n_features)

        Returns
        --------------------------------------------------
            w: ndarray of shape (k, )
            v: ndarray of shape (n_features, k)
        """
        w, v = np.linalg.eig(m)

        return w, v
