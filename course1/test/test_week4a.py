import numpy as np
from numpy.testing import assert_allclose
from nose import tools as nt
from course1 import week4a as w4
from course1.test import case_data as td

np.set_printoptions(precision=10)


def test_initialize():
    # 2 layers
    parameters = w4.initialize_parameters(3, 2, 1)
    nt.eq_(len(parameters), 4)
    nt.eq_(parameters['b2'].shape, (1, 1))
    assert_allclose(parameters['W1'], [
        [0.01624345, -0.00611756, -0.00528172],
        [-0.01072969, 0.00865408, -0.02301539]
    ], rtol=1e-6)
    assert_allclose(parameters['b1'], [[0.], [0.]])
    assert_allclose(parameters['W2'], [[0.01744812, -0.00761207]], rtol=1e-6)
    assert_allclose(parameters['b2'], [[0.]])

    # other layers
    parameters = w4.initialize_parameters_deep(5, 4, 3)
    nt.eq_(len(parameters), 4)
    nt.eq_(parameters['b2'].shape, (3, 1))
    assert_allclose(parameters['W1'], [
        [0.01788628, 0.0043651, 0.00096497, -0.01863493, -0.00277388],
        [-0.00354759, -0.00082741, -0.00627001, -0.00043818, -0.00477218],
        [-0.01313865, 0.00884622, 0.00881318, 0.01709573, 0.00050034],
        [-0.00404677, -0.0054536, -0.01546477, 0.00982367, -0.01101068]
    ], rtol=1e-5)
    assert_allclose(parameters['b1'], [[0.], [0.], [0.], [0.]])
    assert_allclose(parameters['W2'], [
        [-0.01185047, -0.0020565, 0.01486148, 0.00236716],
        [-0.01023785, -0.00712993, 0.00625245, -0.00160513],
        [-0.00768836, -0.00230031, 0.00745056, 0.01976111]
    ], rtol=1e-5)
    assert_allclose(parameters['b2'], [[0.], [0.], [0.]])


def test_linear_forward():
    A, W, b = td.linear_forward_test_case()
    Z, linear_cache = w4.linear_forward(A, W, b)
    assert_allclose(Z, [[3.26295337, -1.23429987]])


def test_linear_activate_forward():
    A_prev, W, b = td.linear_activation_forward_test_case()

    A, linear_activation_cache = w4.linear_activation_forward(A_prev, W, b, activation="sigmoid")
    assert_allclose(A, [[0.96890023, 0.11013289]])

    A, linear_activation_cache = w4.linear_activation_forward(A_prev, W, b, activation="relu")
    assert_allclose(A, [[3.43896131, 0.]])


def test_L_model_forward():
    X, parameters = td.L_model_forward_test_case_2hidden()
    AL, caches = w4.L_model_forward(X, parameters)
    nt.eq_(len(caches), 3)
    assert_allclose(AL, [[0.03921668, 0.70498921, 0.19734387, 0.04728177]])


def test_compute_cost():
    Y, AL = td.compute_cost_test_case()
    cost = w4.compute_cost(AL, Y)
    nt.eq_(cost.shape, ())
    assert_allclose(cost, 0.41493159961539694)


def test_linear_backward():
    dZ, linear_cache = td.linear_backward_test_case()
    dA_prev, dW, db = w4.linear_backward(dZ, linear_cache)
    assert_allclose(dA_prev, [
        [0.51822968, -0.19517421],
        [-0.40506361, 0.15255393],
        [2.37496825, -0.89445391]
    ])
    assert_allclose(dW, [[-0.10076895, 1.40685096, 1.64992505]])
    nt.eq_(db.shape, (1, 1))
    assert_allclose(db, 0.50629448)


def test_linear_activate_backward():
    AL, linear_activation_cache = td.linear_activation_backward_test_case()

    dA_prev, dW, db = w4.linear_activation_backward(AL, linear_activation_cache, "sigmoid")
    assert_allclose(dA_prev, [
        [0.110179936, 0.011053395],
        [0.094668170, 0.009497234],
        [-0.057430922, -0.005761545]
    ])
    assert_allclose(dW, [[0.102667864, 0.09778551, -0.019680842]])
    nt.eq_(db.shape, (1, 1))
    assert_allclose(db, -0.057296222)

    dA_prev, dW, db = w4.linear_activation_backward(AL, linear_activation_cache, "relu")
    assert_allclose(dA_prev, [
        [0.44090989, 0.],
        [0.37883606, 0.],
        [-0.2298228, -0.]
    ])
    assert_allclose(dW, [[0.44513824, 0.37371418, -0.10478989]])
    nt.eq_(db.shape, (1, 1))
    assert_allclose(db, -0.20837892)


def test_L_model_backward():
    AL, Y_assess, caches = td.L_model_backward_test_case()
    grades = w4.L_model_backward(AL, Y_assess, caches)
    assert_allclose(grades['dW1'], [
        [0.41010002, 0.07807203, 0.1379844364, 0.1050216745],
        [0., 0., 0., 0.],
        [0.05283652, 0.0100586544, 0.017777656, 0.0135307956]
    ])
    assert_allclose(grades['db1'], [
        [-0.2200706339], [0.], [-0.02835349]
    ])
    assert_allclose(grades['dA1'], [
        [0.12913162, -0.44014127],
        [-0.14175655, 0.48317296],
        [0.0166370754, -0.0567069755]
    ])


def test_update_parameters():
    parameters, grads = td.update_parameters_test_case()
    parameters = w4.update_parameters(parameters, grads, 0.1)
    assert_allclose(parameters['W1'], [
        [-0.5956206947, -0.09991781, -2.14584584, 1.82662008],
        [-1.7656967649, -0.80627147, 0.5111555653, -1.18258802],
        [-1.0535704, -0.86128581, 0.68284052, 2.2037457748]
    ])
    assert_allclose(parameters['b1'], [
        [-0.04659241], [-1.28888275], [0.53405496]
    ])
    assert_allclose(parameters['W2'], [
        [-0.55569196, 0.0354055, 1.32964895]
    ])
    assert_allclose(parameters['b2'], [[-0.84610769]])
