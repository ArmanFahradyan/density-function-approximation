import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def cubic_spline_interpolation(x, y):
    """ Perform cubic spline interpolation on data (x, y) """
    x = np.array(x)
    if not x.shape:
        x = x.reshape(-1)
    
    n = len(x)
    
    h = np.diff(x)  
    alpha = np.zeros(n)
    
    for i in range(1, n - 1):
        alpha[i] = (3 * (y[i + 1] - y[i]) / h[i]) - (3 * (y[i] - y[i - 1]) / h[i - 1])

    l = np.ones(n)
    mu = np.zeros(n)
    z = np.zeros(n)

    l[0] = 1
    z[0] = 0
    l[n - 1] = 1
    z[n - 1] = 0
    
    for i in range(1, n - 1):
        l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    c = np.zeros(n)
    b = np.zeros(n)
    d = np.zeros(n)
    a = y.copy()

    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
        d[j] = (c[j + 1] - c[j]) / (3 * h[j])
    
    return a, b, c, d, x, h


def evaluate_cubic_spline(a, b, c, d, x, h, x_new):
    """ Evaluate the cubic spline at new points x_new """
    x = np.array(x)
    if not x.shape:
        x = x.reshape(-1)
    x_new = np.array(x_new)
    if not x_new.shape:
        x_new = x_new.reshape(-1)

    n = len(x)
    y_new = np.zeros_like(x_new)

    for i, xi in enumerate(x_new):
        j = np.searchsorted(x, xi) - 1
        j = np.clip(j, 0, n - 2)
        
        dx = xi - x[j]
        y_new[i] = a[j] + b[j] * dx + c[j] * dx**2 + d[j] * dx**3
    
    return y_new


def get_cubic_pdf(data: np.ndarray):
    data_sorted = np.sort(data)

    hist, bin_edges = np.histogram(data_sorted, bins=5, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 
    
    a, b, c, d, x_vals, h_vals = cubic_spline_interpolation(bin_centers, hist)
            
    def get_pdf(x_new):
        return evaluate_cubic_spline(a, b, c, d, x_vals, h_vals, x_new)
            
    return  get_pdf
