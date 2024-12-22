import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from cubic import get_cubic_pdf
from b_splines import get_b_splines_pdf


def get_splines(data: np.ndarray, degree: int, name: str):
        """Estimate the empirical PDF from the data and smooth it using splines."""

        if name == "B-Spline":
            return  get_b_splines_pdf(data, degree )

        elif name == "Cubic":
            return get_cubic_pdf(data, degree)


            



    