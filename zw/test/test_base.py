import numpy as np
from numpy.testing import assert_allclose as assert_eq
from nose import tools as nt
from nose.tools import nottest
from zw.neuralnetwork import NeuralNetwork


@nottest
def test_initialize():
    np.random.seed(1)
    nn = NeuralNetwork(layer_dims=[3, 2, 1], params_ok=True)
    assert_eq(nn.w[0], [
        [0.01624345, -0.00611756, -0.00528172],
        [-0.01072969, 0.00865408, -0.02301539]
    ], rtol=1e-6)
    assert_eq(nn.b[0], [[0], [0]])
    assert_eq(nn.w[1], [[0.01744812, -0.00761207]], rtol=1e-6)
    nt.eq_(nn.b[1].shape, (1, 1))
    assert_eq(nn.b[1], [[0.]])

    np.random.seed(3)
    nn = NeuralNetwork(layer_dims=[5, 4, 3], params_ok=True)
    assert_eq(nn.w[0], [
        [0.01788628, 0.0043651, 0.00096497, -0.01863493, -0.00277388],
        [-0.00354759, -0.00082741, -0.00627001, -0.00043818, -0.00477218],
        [-0.01313865, 0.00884622, 0.00881318, 0.01709573, 0.00050034],
        [-0.00404677, -0.0054536, -0.01546477, 0.00982367, -0.01101068]
    ], rtol=1e-5)
    assert_eq(nn.b[0], [[0.], [0.], [0.], [0.]])
    assert_eq(nn.w[1], [
        [-0.01185047, -0.0020565, 0.01486148, 0.00236716],
        [-0.01023785, -0.00712993, 0.00625245, -0.00160513],
        [-0.00768836, -0.00230031, 0.00745056, 0.01976111]
    ], rtol=1e-5)
    assert_eq(nn.b[1], [[0.], [0.], [0.]])


@nottest
def test_initialize_he():
    np.random.seed(3)
    nn = NeuralNetwork(layer_dims=[2, 4, 1], weight_factor='he', params_ok=True)
    assert_eq(nn.w[0], [
        [1.78862847, 0.43650985],
        [0.09649747, -1.8634927],
        [-0.2773882, -0.35475898],
        [-0.08274148, -0.62700068]
    ], rtol=1e-5)
    assert_eq(nn.w[1], [
        [-0.03098412, -0.33744411, -0.92904268, 0.62552248]
    ], rtol=1e-5)


@nottest
def test_linear_activate_forward():
    np.random.seed(2)
    a_prev = np.random.randn(3, 2)
    w = np.random.randn(1, 3)
    b = np.random.randn(1, 1)

    nn = NeuralNetwork(w=[w], b=[b], params_ok=True)
    a = nn.forward(a_prev, w, b, nn.activation_funcs['sigmoid'][0])
    assert_eq(a, [[0.96890023, 0.11013289]])

    nn = NeuralNetwork(w=[w], b=[b], params_ok=True)
    a = nn.forward(a_prev, w, b, nn.activation_funcs['relu'][0])
    assert_eq(a, [[3.43896131, 0.]])


@nottest
def test_model_forward():
    np.random.seed(6)
    x = np.random.randn(5, 4)
    w1 = np.random.randn(4, 5)
    b1 = np.random.randn(4, 1)
    w2 = np.random.randn(3, 4)
    b2 = np.random.randn(3, 1)
    w3 = np.random.randn(1, 3)
    b3 = np.random.randn(1, 1)

    nn = NeuralNetwork(w=[w1, w2, w3], b=[b1, b2, b3], params_ok=True)
    al = nn.model_forward(x)
    nt.eq_(len(nn.cache), 3)
    assert_eq(al, [[0.03921668, 0.70498921, 0.19734387, 0.04728177]])


@nottest
def test_compute_cost():
    y = np.asarray([[1, 1, 1]])
    al = np.array([[.8, .9, 0.4]])

    nn = NeuralNetwork(layer_dims=(3, 1), params_ok=True)
    cost = nn.compute_cost(al, y)
    assert_eq(cost, 0.41493159961539694)


@nottest
def test_linear_activate_backward():
    np.random.seed(2)
    da = np.random.randn(1, 2)
    a = np.random.randn(3, 2)
    w = np.random.randn(1, 3)
    b = np.random.randn(1, 1)
    z = np.random.randn(1, 2)

    nn = NeuralNetwork(w=[w], b=[b], params_ok=True)
    da_prev, dw, db = nn.backward(da, w, b, (a, z), nn.activation_funcs['sigmoid'][1])
    assert_eq(da_prev, [
        [0.110179936, 0.011053395],
        [0.094668170, 0.009497234],
        [-0.057430922, -0.005761545]
    ])
    assert_eq(dw, [[0.102667864, 0.09778551, -0.019680842]])
    nt.eq_(db.shape, (1, 1))
    assert_eq(db, -0.057296222)

    nn = NeuralNetwork(w=[w], b=[b], params_ok=True)
    da_prev, dw, db = nn.backward(da, w, b, (a, z), nn.activation_funcs['relu'][1])
    assert_eq(da_prev, [
        [0.44090989, 0.],
        [0.37883606, 0.],
        [-0.2298228, -0.]
    ])
    assert_eq(dw, [[0.44513824, 0.37371418, -0.10478989]])
    nt.eq_(db.shape, (1, 1))
    assert_eq(db, -0.20837892)


@nottest
def test_model_backward():
    np.random.seed(3)
    al = np.random.randn(1, 2)
    y = np.array([[1, 0]])

    a1 = np.random.randn(4, 2)
    w1 = np.random.randn(3, 4)
    b1 = np.random.randn(3, 1)
    z1 = np.random.randn(3, 2)

    a2 = np.random.randn(3, 2)
    w2 = np.random.randn(1, 3)
    b2 = np.random.randn(1, 1)
    z2 = np.random.randn(1, 2)

    nn = NeuralNetwork(w=[w1, w2], b=[b1, b2], params_ok=True)
    nn.cache = [(a1, z1), (a2, z2)]
    nn.w = [w1, w2]
    nn.b = [b1, b2]
    grads = nn.model_backward(al, y)

    assert_eq(grads[0][0], [
        [0.41010002, 0.07807203, 0.1379844364, 0.1050216745],
        [0., 0., 0., 0.],
        [0.05283652, 0.0100586544, 0.017777656, 0.0135307956]
    ])
    assert_eq(grads[1][0], [
        [-0.2200706339], [0.], [-0.02835349]
    ])


@nottest
def test_update_parameters():
    np.random.seed(2)
    w1 = np.random.randn(3, 4)
    b1 = np.random.randn(3, 1)
    w2 = np.random.randn(1, 3)
    b2 = np.random.randn(1, 1)

    np.random.seed(3)
    dw1 = np.random.randn(3, 4)
    db1 = np.random.randn(3, 1)
    dw2 = np.random.randn(1, 3)
    db2 = np.random.randn(1, 1)

    nn = NeuralNetwork(w=[w1, w2], b=[b1, b2], learning_rate=0.1, params_ok=True)

    nn.update_parameters(([dw1, dw2], [db1, db2]))

    assert_eq(nn.w[0], [
        [-0.5956206947, -0.09991781, -2.14584584, 1.82662008],
        [-1.7656967649, -0.80627147, 0.5111555653, -1.18258802],
        [-1.0535704, -0.86128581, 0.68284052, 2.2037457748]
    ])
    assert_eq(nn.w[1], [
        [-0.55569196, 0.0354055, 1.32964895]
    ])
    assert_eq(nn.b[0], [
        [-0.04659241], [-1.28888275], [0.53405496]
    ])
    assert_eq(nn.b[1], [[-0.84610769]])
