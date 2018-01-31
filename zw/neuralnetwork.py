import numpy as np
import matplotlib.pyplot as plt
from zw.activation import sigmoid, sigmoid_backward, relu, relu_backward, tanh_backward


class NeuralNetwork:
    activation_funcs = {
        'relu': (relu, relu_backward),
        'sigmoid': (sigmoid, sigmoid_backward),
        'tanh': (np.tanh, tanh_backward)
    }

    def __init__(self, **kargs):
        self.w = kargs.get('w')
        self.b = kargs.get('b')

        layer_dims = kargs.get('layer_dims')
        if layer_dims:
            self.layer_num = len(layer_dims) - 1
        else:
            self.layer_num = len(self.w) or len(self.b)
        if not self.layer_num:
            raise Exception('Invalid arguments: ' % kargs)

        if self.w is None:
            self.w = [np.random.randn(layer_dims[i + 1], layer_dims[i]) * 0.01 for i in range(self.layer_num)]
        if self.b is None:
            self.b = [np.zeros((layer_dims[i + 1], 1)) for i in range(self.layer_num)]

        if 'activations' in kargs:
            avs = kargs['activations']
        else:
            avs = ['relu'] * (self.layer_num - 1) + ['sigmoid']
        self.acfuncs = [self.activation_funcs[a] for a in avs]

        self.cache = []
        self.learning_rate = kargs.get('learning_rate')

    def train(self, x, y, learning_rate=0.0075, num_iterations=3000, callback=None):
        self.learning_rate = learning_rate
        for i in range(0, num_iterations):
            al = self.model_forward(x)
            grads = self.model_backward(al, y)
            self.update_parameters(grads)
            if callback:
                callback(i + 1, self.compute_cost(al, y))

    def predict(self, x, y):
        return self.forward(x) > 0.5

    def update_parameters(self, grads):
        grad_w, grad_b = grads
        self.w = [(wi - gi * self.learning_rate) for wi, gi in zip(self.w, grad_w)]
        self.b = [(bi - gi * self.learning_rate) for bi, gi in zip(self.b, grad_b)]

    def compute_cost(self, a, y):
        cost = -np.sum(y * np.log(a) + (1 - y) * np.log(1 - a), axis=1) / a.shape[1]
        return np.squeeze(cost)

    def model_forward(self, x):
        self.cache = []
        a = x
        for w, b, acfunc in zip(self.w, self.b, self.acfuncs):
            a = self.forward(a, w, b, acfunc[0])
        return a

    def forward(self, a_prev, w, b, acfunc):
        z = w.dot(a_prev) + b
        self.cache.append((a_prev, z))
        return acfunc(z)

    def model_backward(self, al, y):
        grad_w, grad_b = [], []
        dal = -y / al + (1 - y) / (1 - al)
        da = dal
        for i in reversed(range(self.layer_num)):
            da_prev, dw, db = self.backward(da, self.w[i], self.b[i], self.cache[i], self.acfuncs[i][1])
            da = da_prev
            grad_w.append(dw)
            grad_b.append(db)
        grad_w.reverse()
        grad_b.reverse()
        return grad_w, grad_b

    def backward(self, da, w, b, cache, acfunc):
        a_prev, z = cache
        dz = acfunc(da, z)
        m = a_prev.shape[1]
        dw = dz.dot(a_prev.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        da_prev = w.T.dot(dz)
        return da_prev, dw, db


def plot_costs(nn):
    plt.plot(np.squeeze(nn.costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title('Learning rate = ' + str(nn.learning_rate))
    plt.show()
