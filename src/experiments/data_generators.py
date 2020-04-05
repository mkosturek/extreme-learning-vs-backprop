from src.datasets.generators.nested_hyperspheres import generate_nested_hyperspheres
from src.datasets.generators.hypercube_vertices import generate_hypercube_vertices
from src.datasets.generators.spirals \
    import (generate_intertwined_3d_spirals,
            generate_intertwined_archimedean_spirals)

from sklearn.datasets import (make_friedman1,
                              make_friedman2,
                              make_friedman3)

from sklearn.preprocessing import PolynomialFeatures

import numpy as np

from functools import partial


class DataGenerator:

    def generate(self):
        raise NotImplementedError

    def dataset_name(self):
        raise NotImplementedError


class LoaderDataGenerator(DataGenerator):

    def __init__(self, name, loader_fn):
        self.name = name
        self.loader_fn = loader_fn

    def dataset_name(self):
        return self.name

    def generate(self):
        X, y = self.loader_fn()
        return {'name': self.name,
                'data': (X, y.reshape(-1, 1))}


class HypercubeVerticesGenerator(DataGenerator):

    def __init__(self, N, dim, noise=0.1, ensure_linear_nonseparability=True):
        self.N = N
        self.dim = dim
        self.noise = noise
        self.ensure_linear_nonseparability = ensure_linear_nonseparability
        self.name = f'hypercube_vertices_dim_{self.dim}'

    def dataset_name(self):
        return self.name

    def generate(self):
        X, y = generate_hypercube_vertices(self.N, self.dim, self.noise,
                                           self.ensure_linear_nonseparability)
        return {'name': self.name,
                'data': (X, y.reshape(-1, 1))}


class HyperspheresGenerator(DataGenerator):

    def __init__(self, N, dim, center=0, outer_radius=1, noise=0.1):
        self.N = N
        self.dim = dim
        self.center = center
        self.outer_radius = outer_radius
        self.noise = noise
        self.name = f'hyperspheres_dim_{self.dim}'

    def dataset_name(self):
        return self.name

    def generate(self):
        X, y = generate_nested_hyperspheres(self.N, self.dim,
                                            self.center, self.outer_radius,
                                            self.noise)

        return {'name': self.name,
                'data': (X, y.reshape(-1, 1))}


class SpiralGenerator(DataGenerator):

    def __init__(self, N, dim, noise=0.1):

        assert dim in {2, 3}, "Can only generate 2D or 3D spirals."
        self.N = N
        self.dim = dim
        self.noise = noise
        self.name = f'spirals_dim_{self.dim}'

    def dataset_name(self):
        return self.name

    def generate(self):
        if self.dim == 2:
            X, y = generate_intertwined_archimedean_spirals(self.N,
                                                            noise=self.noise)
        else:
            X, y = generate_intertwined_3d_spirals(self.N, noise=self.noise)

        return {'name': self.name,
                'data': (X, y.reshape(-1, 1))}


class Friedman1Generator(DataGenerator):

    def __init__(self, N, dim, noise=0.1):
        self.N = N
        self.dim = dim
        self.noise = noise
        self.name = f'friedman_1_dim_{self.dim}'

    def dataset_name(self):
        return self.name

    def generate(self):
        X, y = make_friedman1(self.N, self.dim, self.noise)
        return {'name': self.name,
                'data': (X, y.reshape(-1, 1))}


class Friedman2Generator(DataGenerator):

    def __init__(self, N, noise=0.1):
        self.N = N
        self.noise = noise
        self.name = 'friedman_2'

    def dataset_name(self):
        return self.name

    def generate(self):
        X, y = make_friedman2(self.N, self.noise)
        return {'name': self.name,
                'data': (X, y.reshape(-1, 1))}


class Friedman3Generator(DataGenerator):

    def __init__(self, N, noise=0.1):
        self.N = N
        self.noise = noise
        self.name = 'friedman_3'

    def dataset_name(self):
        return self.name

    def generate(self):
        X, y = make_friedman3(self.N, self.noise)
        return {'name': self.name,
                'data': (X, y.reshape(-1, 1))}


class FunctionRegressionGenerator(DataGenerator):
    def __init__(self, N, name, func, low_bound=-2 * np.pi, high_bound=2 * np.pi,
                 noise=0.1):
        self.N = N
        self.function = func
        self.low = low_bound
        self.high = high_bound
        self.noise = noise
        self.name = name

    def dataset_name(self):
        return self.name

    def generate(self):
        X = np.random.uniform(self.low, self.high, self.N)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        y = self.function(X) + np.random.normal(0, self.noise, (self.N, 1))
        return {'name': self.name,
                'data': (X, y.reshape(-1, 1))}


SinGenerator = partial(FunctionRegressionGenerator, name='sin', func=np.sin)
CosGenerator = partial(FunctionRegressionGenerator, name='cos', func=np.cos)

SinCGenerator = partial(FunctionRegressionGenerator, name='sinC',
                        func=lambda x: np.sin(x) / x)


class LinearGenerator(DataGenerator):
    def __init__(self, N, dim, low_bound=-2 * np.pi, high_bound=2 * np.pi,
                 noise=0.1):
        self.N = N
        self.dim = dim
        self.low = low_bound
        self.high = high_bound
        self.noise = noise
        self.name = f"linear_dim_{self.dim}"

    def dataset_name(self):
        return self.name

    def generate(self):
        X = np.random.uniform(self.low, self.high, (self.N, self.dim))

        coeffs = np.random.uniform(-5, 5, (self.dim, 1))
        coeffs = coeffs / coeffs.sum()
        bias = np.random.uniform(-5, 5)
        y = X @ coeffs + bias + np.random.normal(0, self.noise, (self.N, 1))

        return {'name': self.name,
                'data': (X, y.reshape(-1, 1))}


class QuadraticGenerator(DataGenerator):
    def __init__(self, N, dim, low_bound=-2 * np.pi, high_bound=2 * np.pi,
                 noise=0.1):
        self.N = N
        self.dim = dim
        self.low = low_bound
        self.high = high_bound
        self.noise = noise
        self.name = f"quadratic_dim_{self.dim}"

    def dataset_name(self):
        return self.name

    def generate(self):
        X = np.random.uniform(self.low, self.high, (self.N, self.dim))
        Xf = PolynomialFeatures(degree=2).fit_transform(X)

        coeffs = np.random.uniform(-5, 5, (Xf.shape[1], 1))
        coeffs = coeffs / coeffs.sum()
        bias = np.random.uniform(-5, 5)
        y = Xf @ coeffs + bias + np.random.normal(0, self.noise, (self.N, 1))

        return {'name': self.name,
                'data': (X, y.reshape(-1, 1))}
