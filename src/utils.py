import numpy as np
from scipy import stats
from scipy.stats import gamma


def normal_pdf(x): return (1 / np.sqrt(2*np.pi)) * np.exp(-x**2 / 2)


def uniform_pdf(x, a=-3.0, b=3.0):
    ans = np.zeros_like(x)
    mask = (a <= x) & (x <= b)
    ans[mask] = 1/(b-a)
    return ans


def gamma_pdf(x, alpha=0.5, lambda_=0.5): return gamma.pdf(x, alpha, scale=lambda_)


def exponential_pdf(x, lambda_=1.0): return lambda_ * np.exp(-lambda_*x) * (x >= 0.0)


def chi_2_pdf(x, df=5): return stats.chi2.pdf(x, df=df)


def normal_generator(n): return np.random.normal(0.0, 1.0, n)


def uniform_generator(n, a=-3.0, b=3.0): return np.random.uniform(a, b, n)


def gamma_generator(n, alpha=0.5, lambda_=0.5): return np.random.gamma(alpha, lambda_, size=n)


def exponential_generator(n, lambda_=1.0): return np.random.exponential(lambda_, n)


def chi_2_generator(n, df=5): return np.random.chisquare(df, n)


distribution_pdf_dict = {
    "normal": normal_pdf,
    "uniform": uniform_pdf,
    "gamma": gamma_pdf,
    "exponential": exponential_pdf,
    "chi_2": chi_2_pdf,
}

distribution_generator_dict = {
    "normal": normal_generator,
    "uniform": uniform_generator,
    "gamma": gamma_generator,
    "exponential": exponential_generator,
    "chi_2": chi_2_generator,
}


def create_data_and_pdf(data_count, distribution="normal"):
    return distribution_generator_dict[distribution](data_count), distribution_pdf_dict[distribution]


def test():
    data, pdf = create_data_and_pdf(data_count=100)


if __name__ == "__main__":
    test()
