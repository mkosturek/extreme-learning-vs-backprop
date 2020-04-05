import numpy as np


def _generate_hypersphere(N, dim, center=0., radius=1.):

    normal_deviates = np.random.normal(size=(N, dim))
    rad = np.sqrt((normal_deviates ** 2).sum(axis=1)).reshape(-1,1)
    points = normal_deviates / rad

    points *= radius
    points += center
    return points


def generate_nested_hyperspheres(N, dim, center=0., outer_radius=1., noise=0.1):

    outer_N = int(N/2)
    inner_N = N - outer_N
    outer_sphere = _generate_hypersphere(outer_N, dim, center, outer_radius)
    outer_sphere += np.random.normal(scale=noise, size=(outer_N, dim))
    outer_labels = np.ones(outer_N, dtype=int)

    inner_sphere = _generate_hypersphere(inner_N, dim, center,
                                         0.5 * outer_radius)
    inner_sphere += np.random.normal(scale=noise, size=(inner_N,dim))
    inner_labels = np.zeros(inner_N, dtype=int)

    perm = np.random.permutation(N)

    X = np.concatenate((outer_sphere, inner_sphere), axis=0)[perm, :]
    y = np.concatenate((outer_labels, inner_labels))[perm]

    return X, y





