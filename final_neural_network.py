import matplotlib.pyplot as plt
import numpy as np

from utils import *

def initialize_parameters(layer_sizes, drop_out_rate):
    parameters = {}

    parameters = {}
    for i in range(1, len(layer_sizes)):

        # L2 Regularization
        W = np.random.randn(layer_sizes[i], layer_sizes[i - 1]) * np.sqrt(2 / layer_sizes[i - 1])
        parameters['W' + str(i)] = W

    return parameters

def initialize_velocity(parameters):

    layers = len(parameters) // 2
    velocity = {}

    for i in range (1,layers + 1):
        velocity['dW' + str(i)] = np.zeros(parameters['W' + str(i)])

    return velocity

def initialize_adams(parameters):

    layers = len(parameters) // 2
    velocity = {}
    step = {}

    for i in range(1, layers + 1):
        velocity['dW' + str(i)] = np.zeros(parameters['W' + str(i)])
        step['dW' + str(i)] = np.zeros(parameters['W' + str(i)])


    return velocity, step


def linear_forward(A, W, gamma, beta, epsilon = 0.999):

    Z = np.dot(W, A)
    m = Z.shape[1]

    # Batch norm
    mean = 1 / m * np.sum(Z, axis=1,keepdims=True)
    variance = 1 / m * np.sum(Z - mean, axis=1, keepdims=True) ** 2
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)
    Z_tilde = gamma * Z_norm + beta

    cache = (A, W)

    return Z_tilde, cache

def sigmoid(Z):

    A = 1/(1+np.exp(-Z))
    cache = Z

    return A, cache


def relu(Z):

    A = np.maximum(0, Z)
    cache = Z

    return A, cache

def forward_propagation(X, parameters, gamma, beta, epsilon=0.999):

    forward_cache = {}
    A = X
    L = len(parameters)

    for i in range(1, L):
        Z_tilde, linear_cache = linear_forward(A, parameters['W' + str(i)], gamma, beta, epsilon)
        A, activation_cache= relu(Z_tilde)

        forward_cache['linear_cache' + str(i)] = linear_cache
        forward_cache['activation_cache' + str(i)] = activation_cache

    ZL, linear_cache = linear_forward(A, parameters['W' + str(L)], gamma, beta, epsilon)
    AL, activation_cache = sigmoid(ZL)

    forward_cache['linear_cache' + str(L)] = linear_cache
    forward_cache['activation_cache' + str(L)] = activation_cache

    return AL, forward_cache

def compute_cost(AL, Y):

    m = Y.shape[1]
    cost = -1/m * np.sum(Y * np.log(AL) + (1-Y) * np.log(1 - AL))

    return cost