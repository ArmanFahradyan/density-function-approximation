import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# # Example distributions
# def make_data_normal():
#     return np.random.normal(0, 1, 1000), lambda x: (1 / np.sqrt(2 * np.pi)) * np.exp(-x**2 / 2)
#
#
# def make_data_binormal():
#     return np.concatenate([np.random.normal(-2, 0.5, 500), np.random.normal(2, 0.5, 500)]), \
#            lambda x: (0.5 * (1 / np.sqrt(2 * np.pi * 0.5**2)) * np.exp(-(x + 2)**2 / (2 * 0.5**2))
#                      + 0.5 * (1 / np.sqrt(2 * np.pi * 0.5**2)) * np.exp(-(x - 2)**2 / (2 * 0.5**2)))
#
#
# def make_data_exp():
#     return np.random.exponential(1, 1000), lambda x: np.exp(-x) * (x >= 0)
#
#
# def make_data_uniform():
#     return np.random.uniform(-2, 2, 1000), lambda x: (np.abs(x) <= 2) / 4


# Kernel Functions
def kernel(name):
    kernels = {
        'gaussian': lambda u: np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi),
        'epanechnikov': lambda u: 0.75 * (1 - u**2) * (np.abs(u) <= 1),
        'cosine': lambda u: (np.pi / 4) * np.cos(np.pi * u / 2) * (np.abs(u) <= 1),
        'linear': lambda u: (1 - np.abs(u)) * (np.abs(u) <= 1)
    }
    return kernels[name]


# Bandwidth Estimation Methods
def bw_silverman(data):
    n = len(data)
    return 1.06 * np.std(data) * n ** (-1 / 5)


def bw_mlcv(data, k):
    n = len(data)

    def log_likelihood(h):
        likelihood = 0
        for i in range(n):
            leave_out = np.delete(data, i)
            kde_i = np.mean(k((data[i] - leave_out) / h)) / h
            if kde_i > 0:
                likelihood += np.log(kde_i)
        return -likelihood

    result = minimize(log_likelihood, 1.0, bounds=[(0.01, 10)], method='L-BFGS-B')
    return result.x[0]


# Kernel Density Estimation
def kde(data, h, k, x):
    x = np.array(x)
    u = (x[:, np.newaxis] - data) / h
    kde_values = np.mean(k(u), axis=1) / h
    return kde_values


kernel_names = ["gaussian", "epanechnikov", "cosine", "linear"]


# Function to create and return the KDE function
def create_kde_function(data, kernel_name, bw_method):

    assert kernel_name in kernel_names, f"Not implemented for kernel {kernel_name}"

    kernel_fn = kernel(kernel_name)

    if bw_method == 'Silverman':
        h = bw_silverman(data)
    elif bw_method == 'MLCV':
        h = bw_mlcv(data, kernel_fn)
    else:
        raise ValueError("Unsupported bandwidth method")

    def kde_function(x):
        return kde(data, h, kernel_fn, x)

    return kde_function


# # Example Usage
# if __name__ == "__main__":
#     data_fn = make_data_normal
#     kernel_name = 'gaussian'
#     bw_method = 'Silverman'
#
#     kde_fn, data, true_dist = create_kde_function(data_fn, kernel_name, bw_method)
#
#     x_plot = np.linspace(np.min(data) * 1.05, np.max(data) * 1.05, 1000)
#     kde_values = kde_fn(x_plot)
#
#     plt.figure(figsize=(8, 6))
#     plt.hist(data, density=True, alpha=0.2, bins=20, rwidth=0.9, label='Histogram')
#     plt.plot(x_plot, kde_values, label='KDE')
#     plt.plot(x_plot, true_dist(x_plot), label='True Distribution', linestyle='--')
#     plt.legend()
#     plt.title(f'KDE with {kernel_name} Kernel and {bw_method} Bandwidth')
#     plt.grid()
#     plt.show()
