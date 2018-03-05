import numpy as np


def binary_crossentropy(A, Y):
    return -Y * np.log(A) - (1 - Y) * np.log(1 - A)


def binary_crossentropy_backprop(A, Y):
    return -Y / A + (1 - Y) / (1 - A)
