import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from os import path as op

from zw.neuralnetwork import NeuralNetwork

#get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

base_dir = op.dirname(__file__)


def train(train_x, train_y, test_x, test_y):
    def print_cost(iteration_num, cost):
        if iteration_num % 100 == 0:
            print("Cost after iteration %d: %f" % (iteration_num, cost))

    nn = NeuralNetwork(layer_dims=[20, 3, 1], num_iterations=3000, weight_factors='deep', callback=print_cost)
    nn.train(train_x, train_y)
    p = nn.predict(train_x)
    print('Accuracy: %f' % np.mean(p == train_y))
    p = nn.predict(test_x)
    print('Accuracy: %f' % np.mean(p == test_y))


if __name__ == '__main__':
    data = scio.loadmat(op.join(base_dir, 'dataset', 'data.mat'))
    train_x = data['X'].T
    train_y = data['y'].T
    test_x = data['Xval'].T
    test_y = data['yval'].T
    plt.scatter(train_x[0, :], train_x[1, :], c=np.squeeze(train_y), s=40, cmap=plt.cm.Spectral)

    np.random.seed(1)
    train(train_x, train_y, test_x, test_y)

    #plt.show()
