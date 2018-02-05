import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def relu(z):
    return np.maximum(0, z)


def sigmoid_backward(da, z):
    a = 1 / (1 + np.exp(-z))
    return da * a * (1 - a)


def relu_backward(da, z):
    # dz = np.array(da, copy=True)
    # dz[z <= 0] = 0
    return da * np.int64(z > 0)


def tanh_backward(da, z):
    return da * (1 - np.square(np.tanh(z)))
