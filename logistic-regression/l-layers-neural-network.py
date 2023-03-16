import numpy as np
from utils import *


def initialize_parameters(layers_size):

    parameters = {}
    for i in range(1,layers_size):
        W = np.random.randn(layers_size[i],layers_size[i - 1]) * 0.01
        b = np.zeros((layers_size, 1))

        parameters['W'+str(i)] = W
        parameters['b'+str(i)] = b

    return parameters


def linear_forward(A, W, b):

    Z = np.dot(W, A) + b
    cache = (A, W, b)

    return Z, cache


def sigmoid(Z):

    A = 1/(1+np.exp(-Z))
    cache = Z

    return A, cache


def relu(Z):

    A = np.max(Z, 0)

    cache = Z

    return A, cache


def linear_forward_activation(A, W, b, L):

    cache = {}

    for i in range(0, L-1):
        Z, linear_cache = linear_forward(A, W, b)
        A, activation_cache = sigmoid(Z)

        cache['linear_cache'+str(i)] = linear_cache
        cache['activation_cache' + str(i)] = activation_cache
        cache['Z'+str(i)] = Z
        cache['A' + str(i)] = A


    ZL, linear_cache = linear_forward(A, W, b)
    AL, activation_cache = sigmoid(ZL)

    cache['ZL'] = ZL

    return AL, cache

def compute_cost(AL, Y):

    m = Y.shape[1]
    cost = -1/m * np.sum(Y * np.log(AL) + (1-Y) * np.log(1 - AL))

    # cost = np.squeeze(cost)

    return cost


def linear_backward(dZ, cache):



if __name__ == '__main__':
    train_set_image, train_set_label, test_set_image, test_set_label = load_dataset()
    train_set_image = reshape(train_set_image)
    test_set_image = reshape(test_set_image)

