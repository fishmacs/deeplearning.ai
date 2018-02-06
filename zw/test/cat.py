import numpy as np
import h5py

from os import path as op

from zw.neuralnetwork import NeuralNetwork

base_dir = op.dirname(__file__)


def load_data():
    train_dataset = h5py.File(op.join(base_dir, 'train_catvnoncat.h5'), "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File(op.join(base_dir, 'test_catvnoncat.h5'), "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def train(train_x, train_y, layers):
    nn = NeuralNetwork(layer_dims=layers)

    def print_cost(iteration_num, cost):
        if iteration_num % 100 == 0:
            print("Cost after iteration %d: %f" % (iteration_num, cost))

    nn.train(train_x, train_y, num_iterations=2500, callback=print_cost)
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
    m = train_x.shape[1]
    print('Accuracy of train set: %f' % (np.sum(p == train_y) / m))

    p = model.predict(test_x)
    m = test_x.shape[1]
    print('Accuracy of test set: %f' % (np.sum(p == test_y) / m))
