import numpy as np
from sklearn.utils import shuffle


def normalise(x):

    x_norm = x - x.min(axis=0)
    denom = x_norm.max(axis=0)
    denom[np.where(denom == 0)] = 1e-9
    x_norm /= denom
    return x_norm


def standardise(x):
    x_std = x - x.mean(axis=0)
    x_std /= x_std.std(axis=0)
    return x_std
