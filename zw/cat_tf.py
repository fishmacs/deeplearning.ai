import numpy as np
import h5py
from os import path as op
import tensorflow as tf
from tf.contrib.layers import xavier_initializer

from zw.neuralnetwork import NeuralNetwork

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

    train_x = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T / 255
    test_x = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T / 255

    return train_x, train_set_y_orig, test_x, test_set_y_orig, classes


def init_one_layer(output_units, input_units, layer_index):
    w = tf.get_variable('W%d' % layer_index, [output_units, input_units], initializer=xavier_initializer())
    b = tf.get_variable('b%d' % layer_index, [output_units, 1], initialiser=tf.zeros_initializer())
    return w, b


def initialize_parameters(*layers):
    return [init_one_layer(layers[i], layers[i-1], i) for i in range(1, len(layers))]

def compute_cost(x, y):
    logits = tf.transpose(x)
    labels = tf.transpose(y)
def train(train_x, train_y, layers):
    nx, m = train_x.shape
    ny = train_y.shape[0]
    parameters = initialize_parameters(nx, *layers)
    #nn = NeuralNetwork(layer_dims=layers, weight_factor=len(layers)>2 and 'deep' or 0.01)

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

    model = train(train_x, train_y, layers)
    p = model.predict(train_x)
    m = train_x.shape[1]
    print('Accuracy of train set: %f' % (np.sum(p == train_y) / m))

    p = model.predict(test_x)
    m = test_x.shape[1]
    print('Accuracy of test set: %f' % (np.sum(p == test_y) / m))
