import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

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

def create_splines(data: np.ndarray, x_fine: np.ndarray, name=str):
    data_sorted = np.sort(data)
    
    n = len(data_sorted)
    x = np.arange(n)
    
    if name == "Cubic":    
        return evaluate_cubic_spline(x_fine, x, data_sorted)
         
    elif name == "Quadratic":    
        return evaluate_quadratic_spline(x_fine, x, data_sorted)

    elif name == "Linear":    
        linear_spline_func = linear_spline(x, data_sorted)
        return linear_spline_func(x_fine)
    else:
        return "Invalid Spline Type"
    
def get_splines_from_data(data: np.ndarray, name: str):
    
    n = len(data)
    x_fine = np.linspace(0, n-1, 1000)

    return create_splines(data,x_fine, name)

def plot_spline(data):
    data_sorted = np.sort(data)

    n = len(data)
    x_fine = np.linspace(0, n-1, 1000)

    xnew = np.arange(len(data))
    tck = interpolate.splrep(xnew, data, s=0)
    ynew = interpolate.splev(x_fine, tck, der=0)

    fig, axs = plt.subplots(1, 4, figsize=(18, 6))

    # Cubic Spline Plot
    axs[0].plot(x_fine, get_splines_from_data(data, "Cubic"), label="Cubic Spline", color='blue')
    axs[0].scatter(np.arange(n), data_sorted, color='black', label="Original Data", alpha=0.5)
    axs[0].set_title("Cubic Spline Fit")
    axs[0].legend()

    # Quadratic Spline Plot
    axs[1].plot(x_fine, get_splines_from_data(data, "Quadratic"), label="Quadratic Spline", color='green')
    axs[1].scatter(np.arange(n), data_sorted, color='black', label="Original Data", alpha=0.5)
    axs[1].set_title("Quadratic Spline Fit")
    axs[1].legend()

    # Linear Spline Plot
    axs[2].plot(x_fine, get_splines_from_data(data, "Linear"), label="Linear Spline", color='red')
    axs[2].scatter(np.arange(n), data_sorted, color='black', label="Original Data", alpha=0.5)
    axs[2].set_title("Linear Spline Fit")
    axs[2].legend()

    axs[3].plot(x_fine, ynew, label="scypy Spline", color='red')
    axs[3].scatter(np.arange(n), data_sorted, color='black', label="Original Data", alpha=0.5)
    axs[3].set_title("Linear Spline Fit")
    axs[3].legend()


    plt.tight_layout()
    plt.savefig("spline_fits.png")

    plt.show()

    return None

data = np.random.normal(0, 1, 400)  
pdf_linear = get_splines_from_data(data, "Cubic")

plot_spline(data)
