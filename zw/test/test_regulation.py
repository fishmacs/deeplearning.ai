import numpy as np
from numpy.testing import assert_allclose as assert_eq
from zw.neuralnetwork import NeuralNetwork

np.set_printoptions(precision=10)


def test_cost_regulation():
    np.random.seed(1)
    y = np.array([[1, 1, 0, 1, 0]])
    w1 = np.random.randn(2, 3)
    b1 = np.random.randn(2, 1)
    w2 = np.random.randn(3, 2)
    b2 = np.random.randn(3, 1)
    w3 = np.random.randn(1, 3)
    b3 = np.random.randn(1, 1)
    a3 = np.array([[0.40682402, 0.01629284, 0.16722898, 0.10118111, 0.40682402]])

    nn = NeuralNetwork(w=[w1, w2, w3], b=[b1, b2, b3], lambd=0.1)
    cost = nn.compute_cost(a3, y)
    assert_eq(cost, 1.78648594516)


def test_backward_propagation_regulation():
    np.random.seed(1)
    x = np.random.randn(3, 5)
    y = np.array([[1, 1, 0, 1, 0]])

    w1 = np.array([
        [-1.09989127, -0.17242821, -0.87785842],
        [0.04221375, 0.58281521, -1.10061918]
    ])
    b1 = np.array([[1.14472371], [0.90159072]])

    w2 = np.array([
        [0.50249434, 0.90085595],
        [-0.68372786, -0.12289023],
        [-0.93576943, -0.26788808]
    ])
    b2 = np.array([[0.53035547], [-0.69166075], [-0.39675353]])

    w3 = np.array([[-0.6871727, -0.84520564, -0.67124613]])
    b3 = np.array([[-0.0126646]])

    z1 = np.array([
        [-1.52855314, 3.32524635, 2.13994541, 2.60700654, -0.75942115],
        [-1.98043538, 4.1600994, 0.79051021, 1.46493512, -0.45506242]
    ])
    a1 = np.array([
        [0., 3.32524635, 2.13994541, 2.60700654, 0.],
        [0., 4.1600994, 0.79051021, 1.46493512, 0.]
    ])

    z2 = np.array([
        [0.53035547, 5.94892323, 2.31780174, 3.16005701, 0.53035547],
        [-0.69166075, -3.47645987, -2.25194702, -2.65416996, -0.69166075],
        [-0.39675353, -4.62285846, -2.61101729, -3.22874921, -0.39675353]
    ])
    a2 = np.array([
        [0.53035547, 5.94892323, 2.31780174, 3.16005701, 0.53035547],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.]
    ])

    z3 = np.array([[-0.3771104, -4.10060224, -1.60539468, -2.18416951, -0.3771104]])
    a3 = np.array([[0.40682402, 0.01629284, 0.16722898, 0.10118111, 0.40682402]])

    nn = NeuralNetwork(w=[w1, w2, w3], b=[b1, b2, b3], lambd=0.7)
    nn.cache = [(x, z1), (a1, z2), (a2, z3)]
    gradw, gradb = nn.model_backward(a3, y)

    assert_eq(gradw[0], [
        [-0.256046467, 0.122988299, -0.28297132],
        [-0.17706304, 0.34536100, -0.4410572]
    ], rtol=1e-5)
    assert_eq(gradw[1], [
        [0.792764876, 0.85133918],
        [-0.0957219, -0.01720463],
        [-0.13100772, -0.03750433]
    ], rtol=1e-5)
    assert_eq(gradw[2], [[-1.77691347, -0.11832879, -0.09397446]], rtol=1e-5)


def test_forward_propagation_dropout():
    np.random.seed(1)
    x = np.random.randn(3, 5)
    w1 = np.random.randn(2, 3)
    b1 = np.random.randn(2, 1)
    w2 = np.random.randn(3, 2)
    b2 = np.random.randn(3, 1)
    w3 = np.random.randn(1, 3)
    b3 = np.random.randn(1, 1)

    np.random.seed(1)
    nn = NeuralNetwork(w=[w1, w2, w3], b=[b1, b2, b3])
    a3 = nn.model_forward(x, keep_prop=0.7)
    assert_eq(a3, [[0.36974721, 0.00305176, 0.04565099, 0.49683389, 0.36974721]], rtol=1e-5)


def test_backward_propagation_dropout():
    np.random.seed(1)

    x = np.random.randn(3, 5)
    y = np.array([[1, 1, 0, 1, 0]])

    w1 = np.array([
        [-1.09989127, -0.17242821, -0.87785842],
        [0.04221375, 0.58281521, -1.10061918]
    ])
    b1 = np.array([[1.14472371], [0.90159072]])

    w2 = np.array([
        [0.50249434, 0.90085595],
        [-0.68372786, -0.12289023],
        [-0.93576943, -0.26788808]
    ])
    b2 = np.array([[0.53035547], [-0.69166075], [-0.39675353]])

    w3 = np.array([[-0.6871727, -0.84520564, -0.67124613]])
    b3 = np.array([[-0.0126646]])

    nn = NeuralNetwork(w=[w1, w2, w3], b=[b1, b2, b3])

    z1 = np.array([
        [-1.52855314, 3.32524635, 2.13994541, 2.60700654, -0.75942115],
        [-1.98043538, 4.1600994, 0.79051021, 1.46493512, -0.45506242]
    ])
    d1 = np.array([
        [True, False, True, True, True],
        [True, True, True, True, False]
    ], dtype=bool)
    a1 = np.array([
        [0., 0., 4.27989081, 5.21401307, 0.],
        [0., 8.32019881, 1.58102041, 2.92987024, 0.]
    ])

    z2 = np.array([
        [0.53035547, 8.02565606, 4.10524802, 5.78975856, 0.53035547],
        [-0.69166075, -1.71413186, -3.81223329, -4.61667916, -0.69166075],
        [-0.39675353, -2.62563561, -4.82528105, -6.0607449, -0.39675353]
    ])
    d2 = np.array([
        [True, False, True, False, True],
        [False, True, False, True, True],
        [False, False, True, False, False]
    ], dtype=bool)
    a2 = np.array([
        [1.06071093, 0., 8.21049603, 0., 1.06071093],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
    ])

    z3 = np.array([[-0.7415562, -0.0126646, -5.65469333, -0.0126646, -0.7415562]])
    a3 = np.array([[0.32266394, 0.49683389, 0.00348883, 0.49683389, 0.32266394]])

    nn.cache = [(x, z1, d1), (a1, z2, d2), (a2, z3)]
    gradw, _ = nn.model_backward(a3, y, keep_prop=0.8)
    assert_eq(gradw[0], [
        [0.0001988393, 0.0002865694, 0.0001213795],
        [0.0003564729, 0.0005137526, 0.0002176053]
    ], rtol=1e-5)
    assert_eq(gradw[1], [
        [-0.0025651848, -0.0009475965],
        [0., 0.],
        [0., 0.]
    ], rtol=1e-5)
    assert_eq(gradw[2], [[-0.0695119105, 0., 0.]])
