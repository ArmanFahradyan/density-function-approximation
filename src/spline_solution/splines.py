import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from src.spline_solution.cubic import get_cubic_pdf
from src.spline_solution.b_splines import get_b_splines_pdf


def get_splines(data: np.ndarray, method: str):
    """Estimate the empirical PDF from the data and smooth it using splines."""

    if method == "B-Spline_1":
        return get_b_splines_pdf(data, 1)
    elif method == "B-Spline_2":
        return get_b_splines_pdf(data, 2)
    elif method == "B-Spline_3":
        return get_b_splines_pdf(data, 3)
    elif method == "Cubic":
        return get_cubic_pdf(data)
