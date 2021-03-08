import numpy as np


def clip(x, q=90):

    x = np.nan_to_num(x)

    min_val = 0
    max_val = np.percentile(x, q, axis=None)

    x = np.clip(x, a_min=min_val, a_max=max_val)
    return x


def standardize(x):

    std = np.std(x)
    median = np.percentile(x, q=50, axis=None)
    return (x - median) / std


def normalize(x):
    min_vol = np.min(x)
    max_vol = np.max(x)

    return (x - min_vol) / (max_vol - min_vol + 1e-3)
