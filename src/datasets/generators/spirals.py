import numpy as np


def _sample_uniformly_distributed_archimedean_thetas(N, start=0, end=3*np.pi):
    thetas = np.random.uniform(0, 1, N)
    # normalisation, following:
    # https://math.stackexchange.com/questions/1371668/equation-to-place-points-equidistantly-on-an-archimedian-spiral-using-arc-length/2216736
    thetas = np.sqrt(2) * np.sqrt(-1 + np.sqrt(1 + 10 ** 2 * thetas ** 2))

    thetas -= thetas.min()
    thetas /= thetas.max()
    thetas *= (end - start)
    thetas += start
    return thetas

def _noise_suppresion(thetas, bias=0.7, dim=3):
    th = thetas - thetas.min()
    th /= th.max()
    return np.repeat((bias + (th * (1-bias))).reshape(-1, 1), 3, axis=1)


def _archimedean_spiral(thetas=np.linspace(0, 1, 1000), a=1, b=0.5):
    x = (a * thetas + b) * np.cos(thetas)
    y = (a * thetas + b) * np.sin(thetas)
    return np.asarray([x, y]).T


def generate_intertwined_archimedean_spirals(N, a=1, b=0.5, start=0, end=3*np.pi, noise=0.1):
    N_1 = int(N/2)
    thetas = _sample_uniformly_distributed_archimedean_thetas(N_1, start, end)
    spiral_1 = _archimedean_spiral(thetas, a, b)

    N_2 = N - N_1
    thetas = _sample_uniformly_distributed_archimedean_thetas(N_2, start, end)
    spiral_2 = -_archimedean_spiral(thetas, a, b)

    X = (np.concatenate([spiral_1, spiral_2], axis=0)
         + np.random.normal(0, noise, (N, 2)))
    y = np.concatenate([np.zeros(N_1, dtype=int), np.ones(N_2, dtype=int)])
    return X, y


def _archimedean_spiral_3d(thetas, a, b, c):
    x = (a * thetas + b) * np.cos(thetas)
    y = (a * thetas + b) * np.sin(thetas)
    z = c * thetas
    return np.asarray([x, y, z]).T


def generate_intertwined_3d_spirals(N, a=1, b=0.5, c=1.2, noise=0.8, length=10*np.pi):
    N_1 = int(N / 2)
    thetas = _sample_uniformly_distributed_archimedean_thetas(N_1, 0, length)
    noise_suppressor =_noise_suppresion(thetas, 1, 3)
    spiral_1 = _archimedean_spiral_3d(thetas, a, b, c)
    spiral_1 += noise_suppressor * np.random.normal(0, noise, (N_1, 3))

    N_2 = N - N_1
    thetas = _sample_uniformly_distributed_archimedean_thetas(N_2, 0, length)
    noise_suppressor =_noise_suppresion(thetas, 1, 3)
    spiral_2 = _archimedean_spiral_3d(thetas, a, np.pi * b * c, c)
    spiral_2 += noise_suppressor * np.random.normal(0, noise, (N_2, 3))
    spiral_2[:, 2] += c * np.pi

    X = np.concatenate([spiral_1, spiral_2], axis=0)
    y = np.concatenate([np.zeros(N_1, dtype=int), np.ones(N_2, dtype=int)])
    return X, y


