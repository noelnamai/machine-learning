#!/usr/bin/env python3

import sys
import random
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr

params = {
    "axes.labelsize"  : 16,
    "xtick.labelsize" : 12,
    "ytick.labelsize" : 12,
    "text.usetex"     : True,
    "font.family"     : "sans-serif"
}

def map_polynomial_features(x, degree = 2):
    """
    Parameters
    --------------------------------------------------
        x     : ndarray of shape (n_samples, n_features)
        degree: int the nth degree polynomial to map features

    Returns
    --------------------------------------------------
        feature_map: ndarray of shape (n_samples, n_features quadratic terms)
    """
    m = x.shape[0]
    n = x.shape[1]
    x = np.hstack((np.ones((m, 1)), x))

    feature_map = list()
    for i in range(m):
        for k, v in enumerate(it.combinations_with_replacement(x[i], degree)):
            if k == 0:
                pass
            else:
                feature_map.append(np.product(v))

    return np.array(feature_map).reshape(m, -1)

def split_train_test(x, y, prop_train = 80, validation = False, seed = None):
    """
    Parameters
    --------------------------------------------------
        x          : ndarray of shape (n_samples, n_features)
        y          : ndarray of shape (n_samples, 1)
        prop_train : (int) percentage to be included in the training set
        random_seed: 

    Returns
    --------------------------------------------------
        x_train:
        x_test :
        y_train:
        y_test :
    """
    np.random.seed(seed = seed)
    if ((x.shape[0] == y.shape[0]) == False):
        raise Exception(f"both x: {x.shape} and y: {y.shape} should be of length n_samples.")
    
    m       = x.shape[0]
    y       = y.reshape(-1, 1)
    index   = list(range(0, m))
    cut_one = int((prop_train * m) / 100)

    np.random.shuffle(index)

    if validation:
        cut_two = int((m - cut_one) / 2) 
        in_train, in_test, in_validation = np.array_split(
            ary = index, 
            indices_or_sections = [cut_one, cut_two]
        )
          
        return x[in_train], x[in_test], x[in_validation], y[in_train], y[in_test], y[in_validation]
    else:
        in_train, in_test = np.array_split(
            ary = index, 
            indices_or_sections = [cut_one]
        )

        return x[in_train], x[in_test], y[in_train], y[in_test]

def plot_cost_function(cost = None, width = 10.0, height = 6.5):
    """
    Parameters
    --------------------------------------------------
        cost  : ndarray of shape (iterations, 1)
        width : float plot width in inches (default: 10.0)
        height: float plot height in inches (default: 6.5)

    Returns
    --------------------------------------------------
        figure: None displays the cost function plot and returns
    """
    plt.rcParams.update(params)
    fig, ax = plt.subplots(figsize = (width, height))
    ax.plot(range(len(cost)), cost, scalex = True, scaley = True)
    ax.axhline(y = min(cost), color = "r", linewidth = 0.5, linestyle = "--")
    ax.set_ylabel("Cost $J(\\theta)$")
    ax.set_xlabel("Number of Iterations $(t)$")
    ax.xaxis.set_major_locator(tkr.MaxNLocator(integer = True))
    ax.margins(0.05)
    ax.axis("tight")
    ax.grid(True)
    fig.tight_layout()

    plt.show()

def mean_squared_error(y_prime, y_test):
    """
    Parameters
    --------------------------------------------------
        y_prime: ndarray of shape (n_samples, 1)
        y_test : ndarray of shape (n_samples, 1)

    Returns
    --------------------------------------------------
        mse: The Mean Squared Error (MSE) 
    """
    m = y_test.shape[0]
    n = y_test.shape[1]
    mse = (1 / m) * np.sum(np.square(y_prime - y_test))
    
    return mse

def root_mean_squared_error(y_prime, y_test):
    """
    Parameters
    --------------------------------------------------
        y_prime: ndarray of shape (n_samples, 1)
        y_test : ndarray of shape (n_samples, 1)

    Returns
    --------------------------------------------------
        rmse: The Root Mean Squared Error (RMSE) 
    """
    rmse = np.sqrt(mean_squared_error(y_prime, y_test))
    
    return rmse
