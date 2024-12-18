import torch
import numpy as np
from matplotlib import pyplot as plt
import argparse

from src.nn_solution.model import CDFEstimator
from src.utils import create_data_and_pdf
from src.nn_solution.train import train
from src.nn_solution.utils import get_model_derivative


def get_pdf_from_data(data: np.ndarray, a: float=-3.0, b: float=3.0):
    data = data.reshape(-1, 1)
    data.sort(axis=0)
    X = torch.tensor(data, dtype=torch.float32)

    model = CDFEstimator(1, [16, 64, 16], 1)
    model = train(model, X, a=a, b=b)
    model.eval()

    empiric_pdf = get_model_derivative(model)

    return empiric_pdf


def main(destination):
    a, b = -3.0, 3.0
    data_count = 200
    data, real_pdf = create_data_and_pdf(data_count)
    data = data.reshape(len(data), 1)
    data.sort(axis=0)
    X = torch.tensor(data, dtype=torch.float32)

    model = CDFEstimator(1, [16, 64, 16], 1)
    model = train(model, X, a=a, b=b)
    model.eval()

    empiric_pdf = get_model_derivative(model)

    x = np.linspace(a, b, 1000).reshape(1000, 1)
    x = torch.tensor(x, dtype=torch.float32)

    answers = empiric_pdf(x).detach().numpy()
    targets = real_pdf(x)

    plt.plot(x.numpy(), targets)
    plt.plot(x.numpy(), answers)
    plt.legend(["target pdf", "empiric pdf"])

    if destination:
        plt.savefig(destination)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--destination", default='')
    args = parser.parse_args()
    main(args.destination)
