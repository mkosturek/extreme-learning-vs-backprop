import numpy as np
from itertools import product
from sklearn.svm import LinearSVC


def _generate_random_vertices(N, dim):
    vertices = np.asarray(list(product([-1, 1], repeat=dim)))
    labels = []
    while len(set(labels)) < 2:
        labels = np.random.choice([0, 1], len(vertices))

    indices = np.random.choice(list(range(len(vertices))), N)
    X = vertices[indices].astype(float)
    y = labels[indices]
    return X, y


def is_linearly_separable(X, y):
    svm = LinearSVC(C=1)
    svm.fit(X, y)
    resp = svm.predict(X)
    return (y == resp).mean() == 1


def generate_hypercube_vertices(N, dim, noise=0.1,
                                ensure_linear_nonseparability=True):
    X, y = _generate_random_vertices(N, dim)

    if ensure_linear_nonseparability:
        while is_linearly_separable(X, y):
            X, y = _generate_random_vertices(N, dim)

    X += np.random.normal(0, noise, (N, dim))
    return X, y
