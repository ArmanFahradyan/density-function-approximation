import numpy as np
import pandas as pd
from scipy.integrate import simps
from scipy.integrate import quad


def L1_distance(f: callable, g: callable, a: float, b: float):
    squared_diff = lambda x: np.abs(f(x) - g(x))

    distance = quad(squared_diff, a, b)[0]

    return distance


def L2_distance(f: callable, g: callable, a: float, b: float):
    squared_diff = lambda x: (f(x) - g(x)) ** 2

    distance = np.sqrt(quad(squared_diff, a, b)[0])

    return distance


def L_inf_distance(f: callable, g: callable, a: float, b: float, node_count: int = 1000):
    x = np.linspace(a, b, node_count)
    f_vals = f(x)
    g_vals = g(x)

    distance = np.max(np.abs(f_vals - g_vals))

    return distance


def cosine_similarity(f: callable, g: callable, a: float, b: float):

    norm_f = np.sqrt(quad(lambda x: f(x)*f(x), a, b)[0])
    norm_g = np.sqrt(quad(lambda x: g(x)*g(x), a, b)[0])

    return quad(lambda x: f(x)*g(x), a, b)[0] / norm_f / norm_g


def compute_all_metrics(f: callable, g: callable, a: float, b: float):
    print("L1 distance:", L1_distance(f, g, a, b))
    print("L2 distance:", L2_distance(f, g, a, b))
    print("L_inf distance:", L_inf_distance(f, g, a, b))
    print("Cosine similarity:", cosine_similarity(f, g, a, b))


def compute_and_store_in_dataframe(target: callable, answers_dict: dict, a: float, b: float):
    df_dict = {}

    for key, answer in answers_dict.items():
        l_1 = L1_distance(target, answer, a, b)
        l_2 = L2_distance(target, answer, a, b)
        l_inf = L_inf_distance(target, answer, a, b)
        cos = cosine_similarity(target, answer, a, b)
        df_dict[key] = [l_1, l_2, l_inf, cos]
    df = pd.DataFrame(df_dict)
    df.index = ["L1", "L2", "L_inf", "Cosine"]
    return df


def test():
    f = lambda x: np.cos(x)
    g = lambda x: np.sin(x)

    print(L2_distance(f, g, 0, np.pi))
    print(L1_distance(f, g, 0, np.pi))
    print(L_inf_distance(f, g, 0, np.pi))
    print(cosine_similarity(f, g, 0, np.pi))


if __name__ == "__main__":
    test()

