
import numpy as np
import matplotlib.pyplot as plt

from src.nn_solution.approximate_pdf import get_pdf_from_data
from src.kernel_solution.kernel_approximation import create_kde_function

from src.utils import create_data_and_pdf
from src.comparison.utils import compute_all_metrics


def main():

    data_count = 200

    distributions = ["normal", "uniform", "exponential", "gamma", "chi_2"]
    kernels = ["gaussian", "epanechnikov", "cosine", "linear"]
    bw_methods = ["Silverman", "MLCV"]
    spline_methods = ["Linear", "Quadratic", "Cubic"]

    fig, axes = plt.subplots(len(distributions), 9, figsize=(60, 48))

    for i, distribution in enumerate(distributions):

        data, real_pdf = create_data_and_pdf(data_count, distribution)
        low_percentile = np.percentile(data, 0.05)
        high_percentile = np.percentile(data, 99.95)
        data = data[(low_percentile <= data) & (data <= high_percentile)]

        # a, b = data.min(), data.max()
        # empiric_std = data.std()
        # print("empiric_std:", empiric_std)
        # a, b = -3.0 * empiric_std, 3.0 * empiric_std
        # a, b = 0.0, 3.0 * empiric_std
        # a, b = -100.0, 100.0
        empiric_pdf = {}
        for kernel in kernels:
            for bw_method in bw_methods:
                empiric_pdf[f"kernel_{kernel}_{bw_method}"] = create_kde_function(data, kernel, bw_method)
        empiric_pdf["nn"] = get_pdf_from_data(data)

        x = np.linspace(-5.0, 5.0, 1000)

        answers = {}
        answers["nn"] = empiric_pdf["nn"](x)
        for kernel in kernels:
            for bw_method in bw_methods:
                answers[f"kernel_{kernel}_{bw_method}"] = empiric_pdf[f"kernel_{kernel}_{bw_method}"](x)
        targets = real_pdf(x)

        for j, (key, value) in enumerate(answers.items()):
            axes[i, j].plot(x, targets)
            axes[i, j].plot(x, value)
            # axes[i, j].legend(["target pdf", f"empiric pdf {key} method"])
            if i + 1 == len(distributions):
                axes[i, j].set_xlabel(f"{key}", fontsize=20)
            if j == 0:
                axes[i, j].set_ylabel(f"{distribution}", fontsize=20)

    destination = "result.png"
    plt.savefig(destination)
    plt.close()

        # compute_all_metrics(real_pdf, empiric_pdf["nn"], -10.0, 10.0)
        # compute_all_metrics(real_pdf, empiric_pdf["kernel"], -10.0, 10.0)


if __name__ == "__main__":
    main()
