import numpy as np

EPS = 1e-6

def softmax(values: np.array):
    values -= np.max(values)
    exp_values = np.exp(values, dtype=np.float64) + EPS
    probs = exp_values / np.sum(exp_values)
    return probs.astype(np.float32)


def gaussian_likelihood(x: np.array, mu: np.array, log_std: np.array):
    pre_sum = -0.5 * (((x - mu) / (np.math.exp(log_std) + EPS)) ** 2 + 2 * log_std + np.math.log(2 * np.pi))
    return np.sum(pre_sum, axis=1)


def polyak_sum(weights: np.array, target_weights: np.array, polyak: float):
    return target_weights * polyak + weights * (1.0 - polyak)


def log_sum_exp(values: np.array, tau=1.0):
    if values.ndim == 1:
        max_value = np.max(values)
        log_sum_exp_values = tau * np.log(np.sum(np.exp((values - max_value) / tau))) + max_value
    else:
        max_value = np.max(values, axis=1, keepdims=True)
        log_sum_exp_values = tau * np.log(np.sum(np.exp((values - max_value) / tau), axis=1, keepdims=True)) + max_value
    return log_sum_exp_values


def random_index(n, k, p=None):
    target_indices = np.random.choice(a=np.arange(n), size=k, p=p, replace=False)
    return target_indices


def top_k_index(a, k):
    target_indices = np.argpartition(a, -k)[-k:]
    return target_indices
