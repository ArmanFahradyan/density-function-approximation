import numpy as np
from scipy import stats
from scipy.stats import gamma


def normal_pdf(x, mu=0.0): return (1 / np.sqrt(2*np.pi)) * np.exp(-(x - mu)**2 / 2)


def two_normal_pdf(x): return 0.75*normal_pdf(x, 0.0) + 0.25*normal_pdf(x, 5.0)


def uniform_pdf(x, a=0.0, b=5.0):
    ans = np.zeros_like(x)
    mask = (a <= x) & (x <= b)
    ans[mask] = 1/(b-a)
    return ans


def exponential_pdf(x, lambda_=1.0): return lambda_ * np.exp(-lambda_*x) * (x >= 0.0)


def chi_2_pdf(x, df=5): return stats.chi2.pdf(x, df=df)


def normal_generator(n, mu=0.0): return np.random.normal(mu, 1.0, n)


def two_normal_generator(n):
    n1 = int(0.75 * n)
    n2 = int(0.25 * n)
    return np.concatenate([normal_generator(n1, 0.0), normal_generator(n2, 5.0)])


def uniform_generator(n, a=0.0, b=5.0): return np.random.uniform(a, b, n)


def exponential_generator(n, lambda_=1.0): return np.random.exponential(lambda_, n)


def chi_2_generator(n, df=5): return np.random.chisquare(df, n)


distribution_pdf_dict = {
    "normal": normal_pdf,
    "two_normal": two_normal_pdf,
    "uniform": uniform_pdf,
    "exponential": exponential_pdf,
    "chi_2": chi_2_pdf,
}

distribution_generator_dict = {
    "normal": normal_generator,
    "two_normal": two_normal_generator,
    "uniform": uniform_generator,
    "exponential": exponential_generator,
    "chi_2": chi_2_generator,
}


def create_data_and_pdf(data_count, distribution="normal"):
    return distribution_generator_dict[distribution](data_count), distribution_pdf_dict[distribution]


def test():
    data, pdf = create_data_and_pdf(data_count=100)


if __name__ == "__main__":
    test()
