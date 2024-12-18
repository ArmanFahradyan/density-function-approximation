
import numpy as np
import matplotlib.pyplot as plt

from src.nn_solution.approximate_pdf import get_pdf_from_data
from src.utils import create_data_and_pdf
from src.comparison.utils import compute_all_metrics


def main():
    destination = "result3.png"
    data_count = 1000

    data, real_pdf = create_data_and_pdf(data_count, "normal")
    low_percentile = np.percentile(data, 0.05)
    high_percentile = np.percentile(data, 99.95)
    data = data[(low_percentile <= data) & (data <= high_percentile)]

    a, b = data.min(), data.max()
    empiric_std = data.std()
    print("empiric_std:", empiric_std)
    a, b = -3.0 * empiric_std, 3.0 * empiric_std
    # a, b = 0.0, 3.0 * empiric_std
    empiric_pdf = get_pdf_from_data(data, a, b)

    x = np.linspace(a, b, 1000)
    x = np.linspace(-10.0, 10.0, 1000)

    answers = empiric_pdf(x)
    targets = real_pdf(x)

    plt.plot(x, targets)
    plt.plot(x, answers)
    plt.legend(["target pdf", "empiric pdf"])

    if destination:
        plt.savefig(destination)
    plt.close()

    compute_all_metrics(real_pdf, empiric_pdf, a, b)


if __name__ == "__main__":
    main()
