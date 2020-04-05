from sklearn.datasets import (load_diabetes as _load_diabetes,
                              load_iris as _load_iris,
                              load_wine as _load_wine,
                              load_breast_cancer as _load_breast_cancer,
                              fetch_california_housing,
                              fetch_covtype)
import numpy as np
import pandas as pd
import cv2
import os
from itertools import chain
from sklearn.preprocessing import LabelEncoder

import torchvision.datasets as tvd
import torchvision.transforms as tvt

from src.datasets.small_norb import SmallNORBDataset
from skimage.feature import hog
import pickle


DATA_PATH = os.path.dirname(os.path.abspath(__file__)) + '/../../data/'


def _load_dataset(name, backup_load_function):
    if (name in os.listdir(DATA_PATH)
            and 'x.npy' in os.listdir(DATA_PATH + name)
            and 'y.npy' in os.listdir(DATA_PATH + name)):
        x = np.load(DATA_PATH + name + '/x.npy')
        y = np.load(DATA_PATH + name + '/y.npy')

    else:
        try:
            os.mkdir(DATA_PATH + name)
        except FileExistsError:
            pass
        x, y = backup_load_function(return_X_y =True)
        np.save(DATA_PATH + name + '/x.npy', x)
        np.save(DATA_PATH + name + '/y.npy', y)
    return x, y

# Regression datasets


def load_california_housing():
    return fetch_california_housing(DATA_PATH, return_X_y=True)


def load_delta_ailerons():
    data = np.genfromtxt(DATA_PATH + '/delta_ailerons/delta_ailerons.data',
                         dtype=np.float32)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y


def load_servo():
    data = pd.read_csv(DATA_PATH + '/servo/servo.data', header=None)
    y = data[4].values
    X = data[[0, 1, 2, 3]]
    X = pd.get_dummies(X, columns=[0, 1, 2, 3]).values.astype(np.float)
    return X, y


def load_machine_cpu():
    data = np.genfromtxt(DATA_PATH + '/machine_cpu/machine.data',
                         delimiter=',', dtype=str)
    X = data[:, 2:-2].astype(np.float32)
    y = data[:, -2].astype(np.float32)
    return X, y


def load_auto_mpg():
    data = pd.read_fwf(DATA_PATH + '/auto_mpg/auto-mpg.data', header=None)
    data = data.apply(lambda col:
                      col.apply(lambda val:
                                val if val != '?' else None))
    data = data.dropna()
    y = data[0].values.astype(float)
    dummies6 = pd.get_dummies(data[6])
    dummies7 = pd.get_dummies(data[7])
    X = pd.concat([data[range(1, 6)], dummies6, dummies7],
                  axis=1).values.astype(float)
    return X, y


def load_bank():
    data1 = np.genfromtxt(DATA_PATH + '/bank/Bank/Bank8FM/bank8FM.data',
                          dtype=np.float32)
    data2 = np.genfromtxt(DATA_PATH + '/bank/Bank/Bank8FM/bank8FM.test',
                          dtype=np.float32)
    data = np.vstack([data1, data2])
    X = data[:, :-1]
    y = data[:, -1]
    return X, y


# Classification datasets

def load_wine():
    return _load_dataset('wine', _load_wine)


def load_iris():
    return _load_dataset('iris', _load_iris)


def load_diabetes():
    return _load_dataset('diabetes', _load_diabetes)


def load_covtype():
    X, y = fetch_covtype(DATA_PATH, return_X_y=True)
    y -= 1
    return X, y


def load_breast_cancer():
    return _load_dataset('breast_cancer', _load_breast_cancer)


def load_urban_land_cover():
    train = pd.read_csv(DATA_PATH + '/urban_land_cover/training.csv')
    test = pd.read_csv(DATA_PATH + '/urban_land_cover/testing.csv')
    df = pd.concat([train, test])
    y = LabelEncoder().fit_transform(df['class'])
    X = df.drop(columns=['class']).values
    return X, y


def load_sonar():
    data = pd.read_csv(DATA_PATH + '/sonar/sonar.all-data', header=None)
    last_column = data.columns[-1]
    y = LabelEncoder().fit_transform(data[last_column].values)
    X = data.drop(columns=last_column).values.astype(np.float32)

    return X, y


def load_glass():
    data = pd.read_csv(DATA_PATH + '/glass_identification/glass.data',
                       header=None, index_col=0)
    last_column = data.columns[-1]
    y = LabelEncoder().fit_transform(data[last_column].values)
    X = data.drop(columns=last_column).values.astype(np.float32)

    return X, y


def load_breast_cancer_ljubljana():
    data = pd.read_csv(DATA_PATH
                       + '/breast_cancer_ljubljana/breast-cancer.data',
                       header=None, na_values='?',
                       names=['Class', 'age', 'menopause', 'tumor-size',
                              'inv-nodes', 'node-caps', 'deg-malig',
                              'breast', 'breast-quad', 'irradiat']).dropna()
    y = LabelEncoder().fit_transform(data['Class'])
    data = data.drop(columns=['Class'])
    X = pd.get_dummies(data, columns=data.columns).values.astype(np.float)

    return X, y


def load_landsat():
    train = np.genfromtxt(DATA_PATH + '/landsat/sat.trn')
    test = np.genfromtxt(DATA_PATH + '/landsat/sat.tst')
    data = np.concatenate([train, test])

    y = LabelEncoder().fit_transform(data[:, -1])
    X = data[:, :-1]
    return X, y


def load_cnae_9():
    data = np.genfromtxt(DATA_PATH + '/cnae-9/CNAE-9.data', delimiter=',')
    y = (data[:, 0] - 1).astype(np.int)
    X = data[:, 1:]
    return X, y


# Images

def load_mnist():
    train_mnist = tvd.MNIST("/mnt/SAMSUNG/datasets/thesis/mnist/",
                            train=True, download=True)
    test_mnist = tvd.MNIST("/mnt/SAMSUNG/datasets/thesis/mnist/",
                           train=False, download=True)

    X = np.concatenate((train_mnist.data,
                        test_mnist.data)) / 255
    y = np.concatenate((train_mnist.targets,
                        test_mnist.targets))
    return X, y


def load_cifar_10():
    train_cifar = tvd.CIFAR10("/mnt/SAMSUNG/datasets/thesis/cifar-10/",
                              train=True, download=True)
    test_cifar = tvd.CIFAR10("/mnt/SAMSUNG/datasets/thesis/cifar-10/",
                             train=False, download=True)

    X = np.concatenate((train_cifar.data,
                        test_cifar.data)) / 255
    y = np.concatenate((train_cifar.targets,
                        test_cifar.targets))
    return X, y


def load_gtsrb(size=(32,32)):
    transform = tvt.Compose([tvt.Resize(size),
                             tvt.Lambda(np.asarray)])
    gtsrb_train = tvd.ImageFolder("/mnt/SAMSUNG/datasets/thesis/GTSRB/Train/",
                                  transform=transform)
    gtsrb_test = tvd.ImageFolder("/mnt/SAMSUNG/datasets/thesis/GTSRB/Test/",
                                 transform=transform)

    X, y = [], []
    for image, label in chain(gtsrb_train, gtsrb_test):
        X.append(image)
        y.append(label)

    X = np.stack(X) / 255
    y = np.asarray(y)
    return X, y


def load_norb(size=(96,96)):
    dataset = SmallNORBDataset("/mnt/SAMSUNG/datasets/thesis/smallnorb/")
    X, y = [], []
    for example in chain(dataset.data["train"],
                                   dataset.data["test"]):
        X.append(cv2.resize(example.image_lt / 255, size))
        y.append(example.category)
        # X.append(example.image_rt / 255)
        # y.append(example.category)

    X = np.stack(X)
    y = np.asarray(y)
    return X, y


def _hog_extractor(X, orientations=9,
                   pixels_per_cell=(8, 8),
                   cells_per_block=(3, 3)):
    X = (X * 255).astype(np.uint8)
    features = []
    for image in X:
        descr = hog(image,
                    orientations=orientations,
                    pixels_per_cell=pixels_per_cell,
                    cells_per_block=cells_per_block,
                    visualize=False,
                    multichannel=len(image.shape) == 3)
        features.append(descr)
    X = np.stack(features)
    return X


def load_hog_mnist():
    if "hog_dataset.pkl" in os.listdir("/mnt/SAMSUNG/datasets/thesis/mnist/"):
        with open("/mnt/SAMSUNG/datasets/mnist/hog_dataset.pkl", "rb") as f:
            X, y = pickle.load(f)
    else:
        X, y = load_mnist()
        X = _hog_extractor(X)
    return X, y


def load_hog_cifar_10():
    if "hog_dataset.pkl" in os.listdir("/mnt/SAMSUNG/datasets/thesis/cifar-10/"):
        with open("/mnt/SAMSUNG/datasets/cifar-10/hog_dataset.pkl", "rb") as f:
            X, y = pickle.load(f)
    else:
        X, y = load_cifar_10()
        X = _hog_extractor(X)
    return X, y


def load_hog_gtsrb():
    if "hog_dataset.pkl" in os.listdir("/mnt/SAMSUNG/datasets/thesis/GTSRB/"):
        with open("/mnt/SAMSUNG/datasets/GTSRB/hog_dataset.pkl", "rb") as f:
            X, y = pickle.load(f)
    else:
        X, y = load_gtsrb()
        X = _hog_extractor(X)
    return X, y


def load_hog_norb():
    if "hog_dataset.pkl" in os.listdir("/mnt/SAMSUNG/datasets/thesis/smallnorb/"):
        with open("/mnt/SAMSUNG/datasets/smallnorb/hog_dataset.pkl", "rb") as f:
            X, y = pickle.load(f)
    else:
        X, y = load_norb()
        X = _hog_extractor(X)
    return X, y
