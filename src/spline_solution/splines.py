import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def basis_function(t, i, k, x):
    """Recursive definition of B-spline basis function."""
    if k == 0:
        return 1.0 if t[i] <= x < t[i + 1] else 0.0
    else:
        left = ((x - t[i]) / (t[i + k] - t[i])) * basis_function(t, i, k-1, x) if (t[i + k] > t[i]) else 0.0
        right = ((t[i + k + 1] - x) / (t[i + k + 1] - t[i + 1])) * basis_function(t, i + 1, k-1, x) if (t[i + k + 1] > t[i + 1]) else 0.0
        return left + right

def spline_fit(x, y, degree=3):
    """Fit a B-spline manually to data."""
    n = len(x)
    t = np.concatenate((np.repeat(x[0], degree), x, np.repeat(x[-1], degree))) 
    m = len(t) - degree - 1 
    B = np.zeros((n, m))  
    
    for i in range(m):
        for j in range(n):
            B[j, i] = basis_function(t, i, degree, x[j])
    
    coeffs = np.linalg.lstsq(B, y, rcond=None)[0]
    return t, coeffs

def evaluate_spline(t, coeffs, degree, x_new):
    """Evaluate a fitted B-spline at new points."""
    n = len(x_new)
    m = len(coeffs)
    y_new = np.zeros(n)
    
    for j in range(n):
        for i in range(m):
            y_new[j] += coeffs[i] * basis_function(t, i, degree, x_new[j])
    
    return y_new

def get_splines(data: np.ndarray, degree: int):
        """Estimate the empirical PDF from the data and smooth it using B-splines."""
        data_sorted = np.sort(data)

        hist, bin_edges = np.histogram(data_sorted, bins=5, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  

        t, coeffs = spline_fit(bin_centers, hist, degree)
        
        def get_pdf(x_new):
            return evaluate_spline(t, coeffs, degree, x_new)
        
        return  get_pdf



    