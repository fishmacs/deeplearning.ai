import numpy as np
from numpy.testing import assert_allclose as assert_eq

from zw.nn import NeuralNetwork, DenseLayer


def test_layer_forward():
    np.random.seed(2)

    Aprev = np.random.randn(3, 2)
    W = np.random.randn(1, 3)
    b = np.random.randn(1, 1)

    layer = DenseLayer(W, b)
    A = layer.forward(Aprev)
    assert_eq(A, [[3.43896131, 0.]])

    layer = DenseLayer(W, b, activation='sigmoid')
    A = layer.forward(Aprev)
    assert_eq(A, [[0.96890023, 0.11013289]])


def test_layer_backward():
    np.random.seed(2)

    dA = np.random.randn(1, 2)
    A = np.random.randn(3, 2)
    W = np.random.randn(1, 3)
    b = np.random.randn(1, 1)
    Z = np.random.randn(1, 2)
    layer = DenseLayer(np.array(W), np.array(b))
    layer.Aprev = A
    layer.Z = Z
    dAprev = layer.backward(dA, learning_rate=0.0075)
    assert_eq(dAprev, [
        [0.44090989, 0.],
        [0.37883606, 0.],
        [-0.2298228, -0.]
    ])
    assert_eq(layer.W, [[-1.06129076, -0.91181047, 0.55223997]])
    assert_eq(layer.b, [[2.29377085]])

    layer = DenseLayer(np.array(W), np.array(b), activation='sigmoid')
    layer.Aprev = A
    layer.Z = Z
    dAprev = layer.backward(dA, learning_rate=0.0075)
    assert_eq(dAprev, [
        [0.110179936, 0.011053395],
        [0.094668170, 0.009497234],
        [-0.057430922, -0.005761545]
    ])
    assert_eq(layer.W, [[-1.05872223, -0.90974101, 0.55160165]])
    assert_eq(layer.b, [[2.29263773]])


def test_forward():
    np.random.seed(6)

    nn = NeuralNetwork(lostfunc='binary_crossentropy')

    X = np.random.randn(5, 4)

    W = np.random.randn(4, 5)
    b = np.random.randn(4, 1)
    nn.add_layer(DenseLayer(W, b))

    W = np.random.randn(3, 4)
    b = np.random.randn(3, 1)
    nn.add_layer(DenseLayer(W, b))

    W = np.random.randn(1, 3)
    b = np.random.randn(1, 1)
    nn.add_layer(DenseLayer(W, b, activation='sigmoid'))

    A = nn.forward(X)
    assert_eq(A, [[0.03921668, 0.70498921, 0.19734387, 0.04728177]])


def test_compute_cost():
    Y = np.asarray([[1, 1, 1]])
    A = np.asarray([[.8, .9, .4]])

    nn = NeuralNetwork(lostfunc='binary_crossentropy')
    cost = nn.compute_cost(A, Y)
    assert_eq(cost, 0.41493160)
