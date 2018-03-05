import numpy as np
import h5py

from os import path as op

from zw.nn import NeuralNetwork, DenseLayer

base_dir = op.dirname(__file__)


def load_data():
    train_dataset = h5py.File(op.join(base_dir, 'dataset', 'train_catvnoncat.h5'), "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File(op.join(base_dir, 'dataset', 'test_catvnoncat.h5'), "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def train(train_x, train_y, layers):
    nn = NeuralNetwork(lostfunc='binary_crossentropy')
    layers.insert(0, train_x.shape[0])
    for (m, n) in zip(layers, layers[1:]):
        W = np.random.randn(n, m) / np.sqrt(m)
        b = np.zeros((n, 1))
        nn.add_layer(DenseLayer(W, b))
    nn.layers[-1].set_activation('sigmoid')
    nn.train(train_x, train_y, num_iterations=2500, learning_rate=0.0075, print_cost=100)
    return nn


if __name__ == '__main__':
    np.random.seed(1)

    import sys
    layers = [int(n) for n in sys.argv[1:]]

    train_x, train_y, test_x, test_y, classes = load_data()
    train_x = train_x.reshape(train_x.shape[0], -1).T / 255
    test_x = test_x.reshape(test_x.shape[0], -1).T / 255

    model = train(train_x, train_y, layers)
    p = model.predict(train_x)
    p = p > 0.5
    m = train_x.shape[1]
    print('Accuracy of train set: %f' % (np.sum(p == train_y) / m))

    p = model.predict(test_x)
    p = p > 0.5
    m = test_x.shape[1]
    print('Accuracy of test set: %f' % (np.sum(p == test_y) / m))
