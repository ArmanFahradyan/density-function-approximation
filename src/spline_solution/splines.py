import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy.stats as stats
from scipy.interpolate import interp1d

def normalize_pdf(pdf, x_fine):
    area = np.trapz(pdf, x_fine)
    return pdf / area

def make_data_normal():
    return np.random.normal(0, 1, 1000), lambda x: (1 / np.sqrt(2 * np.pi)) * np.exp(-x**2 / 2)


def cubic_spline(x, y):
    n = len(x)
    h = np.diff(x)

    A = np.zeros((n, n)) 
    b = np.zeros(n)

    A[0, 0] = 1
    A[n-1, n-1] = 1
    b[0] = 0
    b[n-1] = 0

    for i in range(1, n-1):
        A[i, i-1] = h[i-1]
        A[i, i] = 2 * (h[i-1] + h[i])
        A[i, i+1] = h[i]
        b[i] = 6 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])

    M = np.linalg.solve(A, b)

    m = np.zeros(n-1)
    for i in range(n-1):
        m[i] = (y[i+1] - y[i]) / h[i] - h[i] * (2 * M[i] + M[i+1]) / 6

    return M, m

def evaluate_cubic_spline(x_data, x, y):
    n = len(x) - 1
    y_interpolated = np.zeros(len(x_data))
    M, m = cubic_spline(x, y)

    for i in range(n-1):
        mask = (x_data >= x[i]) & (x_data <= x[i+1])
        h = x_data[mask] - x[i]
        y_interpolated[mask] = y[i] + m[i] * h + M[i] * h**2 / 2 + (M[i+1] - M[i]) * h**3 / 6

    return y_interpolated

def quadratic_spline(x, y):
    n = len(x)  
    d_0 = 0

    A = np.zeros((3, n))
    d = np.zeros(n)

    d[0] = d_0
    d[n-1] = 0

    for i in range(0, n-1):
        A[2, i] = (y[i+1] - y[i]) - d[i]
        A[1, i] = -2*x[i]* A[2,i] + d[i]
        A[0, i] = -x[i]*(x[i] * A[2,i] + A[1,i]) + y[i]
        d[i+1] = 2*A[2,i]*x[i+1] + A[1,i]

    return A

def evaluate_quadratic_spline(x_data, x, y):
    n = len(x) - 1
    y_interpolated = np.zeros(len(x_data))
    A = quadratic_spline(x, y)

    for i in range(n-1):
        mask = (x_data >= x[i]) & (x_data <= x[i+1])
        h = x_data[mask] - x[i]
        y_interpolated[mask] = A[2,i]*x_data[mask]**2 + A[1,i]*x_data[mask]  + A[0,i]
    
    return y_interpolated

def linear_spline(x, y):
    def linear_interpolation(x_data):
        y_interpolated = np.zeros(len(x_data))
        for i in range(len(x_data)):
            for j in range(len(x)-1):
                if x[j] <= x_data[i] <= x[j+1]:
                    y_interpolated[i] = y[j] + (y[j+1] - y[j]) * (x_data[i] - x[j]) / (x[j+1] - x[j])
        return y_interpolated
    
    return linear_interpolation

def create_splines(data: np.ndarray, x_fine: np.ndarray, x:np.ndarray, name=str):
    
    n = len(data)
    
    if name == "Cubic":    
        return evaluate_cubic_spline(x_fine, x, data)
         
    elif name == "Quadratic":    
        return evaluate_quadratic_spline(x_fine, x, data)

    elif name == "Linear":    
        linear_spline_func = linear_spline(x, data)
        return linear_spline_func(x_fine)
    else:
        return "Invalid Spline Type"
    
def get_splines_from_data(data: np.ndarray, name: str):

    hist, bin_edges = np.histogram(data, bins=10, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    x_fine = np.linspace(min(bin_edges), max(bin_edges), 100)

    pdf_linear = create_splines(hist, name, x_fine, bin_centers)
    normalized_pdf = normalize_pdf(pdf_linear, x_fine)
    
    return normalized_pdf
