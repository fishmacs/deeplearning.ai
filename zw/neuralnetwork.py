import numpy as np
import matplotlib.pyplot as plt
from zw.activation import sigmoid, sigmoid_backward, relu, relu_backward, tanh_backward


def initial_weights(units, last_units, weight_factor):
    w = np.random.randn(units, last_units)
    if isinstance(weight_factor, float):
        w *= weight_factor
    elif weight_factor == 'he':
        w *= np.sqrt(2 / last_units)
    else:
        raise Exception('Unknown weight factor: ' + weight_factor)
    return w


class NeuralNetwork:
    activation_funcs = {
        'relu': (relu, relu_backward),
        'sigmoid': (sigmoid, sigmoid_backward),
        'tanh': (np.tanh, tanh_backward)
    }

    def __init__(self, **kargs):
        self.params = kargs

    def __getattr__(self, key):
        if key in self.params:
            return self.params[key]
        return None

    def init_params(self, params):
        self.w = params.get('w')
        self.b = params.get('b')

        layer_dims = params.get('layer_dims')
        if layer_dims:
            self.layer_num = len(layer_dims) - 1
        else:
            self.layer_num = len(self.w) or len(self.b)
        if not self.layer_num:
            raise Exception('Invalid arguments: ' % params)

        weight_factor = params.get('weight_factor', 0.01)
        if self.w is None:
            self.w = [initial_weights(layer_dims[i + 1], layer_dims[i], weight_factor) for i in range(self.layer_num)]
        if self.b is None:
            self.b = [np.zeros((layer_dims[i + 1], 1)) for i in range(self.layer_num)]

        if 'activations' in params:
            avs = params['activations']
        else:
            avs = ['relu'] * (self.layer_num - 1) + ['sigmoid']
        self.acfuncs = [self.activation_funcs[a] for a in avs]

        self.cache = []
        self.learning_rate = params.get('learning_rate', 0.0075)
        self.lambd = params.get('lambd', 0)

    def train(self, x, y, **kargs):  # learning_rate=0.0075, num_iterations=3000, callback=None, lambd=0, keep_prop=1.)
        self.params.update(kargs)
        self.init_params(self.params)
        self.learning_rate = learning_rate
        self.lambd = lambd
        for i in range(0, num_iterations):
            al = self.model_forward(x, keep_prop)
            if callback:
                callback(i + 1, self.compute_cost(al, y))
            grads = self.model_backward(al, y, keep_prop)
            self.update_parameters(grads)

    def predict(self, x):
        return self.model_forward(x) > 0.5

    def update_parameters(self, grads):
        grad_w, grad_b = grads
        self.w = [(wi - gi * self.learning_rate) for wi, gi in zip(self.w, grad_w)]
        self.b = [(bi - gi * self.learning_rate) for bi, gi in zip(self.b, grad_b)]

    def compute_cost(self, a, y):
        cost = -np.sum(y * np.log(a) + (1 - y) * np.log(1 - a), axis=1) / a.shape[1]
        m = a.shape[1]
        if self.lambd:
            cost += self.lambd * sum([np.sum(np.square(w)) for w in self.w]) / (2 * m)
        return np.squeeze(cost)

    def model_forward(self, x, keep_prop=1.):
        self.cache = []
        a = x
        if isinstance(keep_prop, float):
            keep_prop = [keep_prop] * self.layer_num
            keep_prop[-1] = 1
        for w, b, acfunc, keeprop in zip(self.w, self.b, self.acfuncs, keep_prop):
            a = self.forward(a, w, b, acfunc[0], keeprop)
        return a

    def forward(self, a_prev, w, b, acfunc, keeprop=1):
        z = w.dot(a_prev) + b
        cache = [a_prev, z]
        a = acfunc(z)
        if keeprop < 1:
            d = np.random.rand(*a.shape) < keeprop
            a *= d
            a /= keeprop
            cache.append(d)
        self.cache.append(cache)
        return a

    def model_backward(self, al, y, keep_prop=1.):
        grad_w, grad_b = [], []
        dal = -y / al + (1 - y) / (1 - al)
        da = dal
        if isinstance(keep_prop, float):
            keep_prop = [keep_prop] * self.layer_num
            keep_prop[-1] = 1
        for i in reversed(range(self.layer_num)):
            da_prev, dw, db = self.backward(da, self.w[i], self.b[i], self.cache[i], self.acfuncs[i][1], keep_prop[i])
            da = da_prev
            grad_w.append(dw)
            grad_b.append(db)
        grad_w.reverse()
        grad_b.reverse()
        return grad_w, grad_b

    def backward(self, da, w, b, cache, acfunc, keeprop=1):
        if keeprop < 1:
            a_prev, z, d = cache
            da *= d
            da /= keeprop
        else:
            a_prev, z = cache
        dz = acfunc(da, z)
        m = a_prev.shape[1]
        dw = dz.dot(a_prev.T) / m
        if self.lambd:
            dw += self.lambd * w / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        da_prev = w.T.dot(dz)
        return da_prev, dw, db


def plot_costs(nn):
    plt.plot(np.squeeze(nn.costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title('Learning rate = ' + str(nn.learning_rate))
    plt.show()
