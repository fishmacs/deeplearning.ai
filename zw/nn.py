import numpy as np
import zw.activation as actv
import zw.cost as cost

actvkeys = ['relu', 'sigmoid', 'tanh']
actvfuncs = {k: (getattr(actv, k), getattr(actv, k + '_backward')) for k in actvkeys}

lostkeys = ['binary_crossentropy']
lostfuncs = {k: (getattr(cost, k), getattr(cost, k + '_backprop')) for k in lostkeys}


class NeuralNetwork:
    def __init__(self, lostfunc):
        self.layers = []
        # self.hyperparams = {}
        self.lostfunc, self.lostback = lostfuncs[lostfunc]

    def add_layer(self, layer):
        self.layers.append(layer)

    def train(self, X, Y, **kargs):
        print_cost = kargs['print_cost']
        learning_rate = kargs['learning_rate']

        for i in range(kargs['num_iterations']):
            A = self.forward(X)
            self.backward(A, Y, learning_rate)

            if print_cost > 0 and i % print_cost == 0:
                print('Cost after iteration %d: %f' % (i, self.compute_cost(A, Y)))

    def forward(self, X):
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def backward(self, A, Y, learning_rate):
        dA = self.lostback(A, Y)
        for i in range(len(self.layers), 0, -1):
            dA = self.layers[i - 1].backward(dA, learning_rate)

    def predict(self, X):
        return self.forward(X)

    def compute_cost(self, A, Y):
        m = A.shape[1]
        return np.sum(self.lostfunc(A, Y), axis=1) / m


class NNlayer:
    def __init__(self, activation):
        self.set_activation(activation)

    def set_activation(self, activation):
        self.activation, self.actvback = actvfuncs[activation]

    def forward(self, Aprev):
        raise Exception('Unimplemented!')

    def backward(self, dA):
        raise Exception('Unimplemented')


class DenseLayer(NNlayer):
    def __init__(self, W, b, activation='relu', name='dense'):
        super().__init__(activation)
        self.W = W
        self.b = b
        self.name = name

    def forward(self, Aprev):
        Z = self.W.dot(Aprev) + self.b
        self.A = self.activation(Z)
        self.Z = Z
        self.Aprev = Aprev
        return self.A

    def backward(self, dA, learning_rate):
        dZ = self.actvback(dA, self.Z)
        m = self.Aprev.shape[1]
        dW = dZ.dot(self.Aprev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dAprev = self.W.T.dot(dZ)
        self.W -= dW * learning_rate
        self.b -= db * learning_rate
        return dAprev
