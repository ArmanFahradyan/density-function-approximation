
import os
import numpy as np
import matplotlib.pyplot as plt

from src.nn_solution.approximate_pdf import get_pdf_from_data
from src.kernel_solution.kernel_approximation import create_kde_function
from src.spline_solution.splines import get_splines

from src.utils import create_data_and_pdf
from src.comparison.utils import compute_all_metrics, compute_and_store_in_dataframe


def main():

    data_count = 200

    distributions = ["normal", "two_normal", "uniform", "exponential", "chi_2"]
    distributions_dict = {"normal": "Gaussian(0, 1)",
                          "two_normal": "0.75*N(0, 1)+0.25*N(5, 1)",
                          "uniform": "Uniform(0, 5)",
                          "exponential": "Exp(1)",
                          "chi_2": "Chi^2(5)"}

    kernels = ["gaussian", "epanechnikov", "cosine", "linear"]
    bw_methods = ["Silverman", "MLCV"]
    spline_methods = ["B-Spline_1", "B-Spline_2", "B-Spline_3"]

    fig, axes = plt.subplots(len(distributions), 1+8+3, figsize=(80, 64))

    os.makedirs("metrics", exist_ok=True)

    for i, distribution in enumerate(distributions):

        data, real_pdf = create_data_and_pdf(data_count, distribution)
        low_percentile = np.percentile(data, 0.05)
        high_percentile = np.percentile(data, 99.95)
        data = data[(low_percentile <= data) & (data <= high_percentile)]

        empiric_pdf = {}
        for spline_method in spline_methods:
            empiric_pdf[f"spline_{spline_method}"] = get_splines(data, spline_method)
        for kernel in kernels:
            for bw_method in bw_methods:
                empiric_pdf[f"kernel_{kernel}_{bw_method}"] = create_kde_function(data, kernel, bw_method)
        empiric_pdf["nn"] = get_pdf_from_data(data)

        x = np.linspace(-3.0, 7.0, 1000)

        answers = {}
        for spline_method in spline_methods:
            answers[f"spline_{spline_method}"] = empiric_pdf[f"spline_{spline_method}"](x)

        answers["nn"] = empiric_pdf["nn"](x)

        for kernel in kernels:
            for bw_method in bw_methods:
                answers[f"kernel_{kernel}_{bw_method}"] = empiric_pdf[f"kernel_{kernel}_{bw_method}"](x)
        targets = real_pdf(x)

        for j, (key, value) in enumerate(answers.items()):
            axes[i, j].plot(x, targets)
            axes[i, j].plot(x, value)
            if i + 1 == len(distributions):
                axes[i, j].set_xlabel(f"{key}", fontsize=20)
            if j == 0:
                axes[i, j].set_ylabel(f"{distributions_dict[distribution]}", fontsize=20)

        df = compute_and_store_in_dataframe(real_pdf, empiric_pdf, -7.0, 7.0)
        df.to_csv(os.path.join("metrics", f"{distribution}.csv"))

    destination = "result.png"
    plt.savefig(destination)
    plt.close()


if __name__ == "__main__":
    main()
