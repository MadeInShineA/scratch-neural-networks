import numpy as np
import h5py

def initialize_parameters(x_size,hidden_size,output_size):

    W1 = np.random.randn(hidden_size, x_size) * 0.01
    b1 = np.zeros(hidden_size, 1)
    W2 = np.random.randn(output_size, hidden_size) * 0.01
    b2 = np.zeros(output_size, 1)

    return W1, b1, W2, b2

def linear_forward (A, W, b):

    Z = np.dot(W,A) + b
    cache = (A, W, b)

    return Z, cache

def sigmoid(Z):

    A = 1/(1+np.exp(-Z))
    cache = Z

    return A, cache

def relu(Z):

    A = np.maximum(0, Z)
    cache = Z

    return A, cache

def linear_activation_forward(previous_A, W, b, activation):

    Z, linear_cache = linear_forward(previous_A, W, b)
    if activation =="sigmoid":
        A, activation_cache = sigmoid(Z)

    if activation =="relu":
        A, activation_cache = relu(Z)

    cache = linear_cache, activation_cache

    return A, cache

def compute_cost(AL, Y):

    m = Y.shape[1]
    cost = -1/m * np.sum(Y * np.log(AL) + (1-Y) * np.log(1-AL))

    # cost = np.squeeze(cost)

    return cost

def linear_backward(dZ, cache):

    previous_A, W, b = cache
    m = np.shape(previous_A)[1]

    dW = 1/m * np.dot(dZ, previous_A.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    dprevious_A = np.dot(W.T, dZ)

    return dprevious_A, dW, db

def sigmoid_backward(dA, cache):

    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)

    return dZ

def relu_backward(dA, cache):

    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0

    return dZ
def linear_activation_backward(dA, cache, activation):

    linear_cache, activation_cache = cache

    if activation == "signoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dprevious_A, dW, db = linear_backward(dZ, linear_cache)

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dprevious_A, dW, db = linear_backward(dZ, linear_cache)

    return dprevious_A, dW, db

def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2,  learning_rate):

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1

    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    return W1, b1, W2, b2






