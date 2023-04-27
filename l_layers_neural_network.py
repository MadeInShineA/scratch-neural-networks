import matplotlib.pyplot as plt
import numpy as np

from utils import *


def initialize_parameters(layers_sizes):

    parameters = {}
    for i in range(1,len(layers_sizes)):
        W = np.random.randn(layers_sizes[i],layers_sizes[i - 1]) * np.sqrt(2 / layers_sizes[i-1])
        b = np.zeros((layers_sizes[i], 1))

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

    A = np.maximum(0, Z)
    cache = Z

    return A, cache



def linear_activation_forward(X, parameters):

    forward_cache = {}
    A = X
    L = len(parameters) // 2

    for i in range(1, L):
        Z, linear_cache = linear_forward(A, parameters['W'+str(i)], parameters['b'+str(i)])
        A, activation_cache = relu(Z)

        forward_cache['linear_cache'+str(i)] = linear_cache
        forward_cache['activation_cache' + str(i)] = activation_cache

    ZL, linear_cache = linear_forward(A, parameters['W'+str(L)], parameters['b'+str(L)])
    AL, activation_cache = sigmoid(ZL)

    forward_cache['linear_cache'+str(L)] = linear_cache
    forward_cache['activation_cache'+str(L)] = activation_cache


    return AL, forward_cache

def compute_cost(AL, Y):

    m = Y.shape[1]
    cost = -1/m * np.sum(Y * np.log(AL) + (1-Y) * np.log(1 - AL))

    # cost = np.squeeze(cost)

    return cost


def linear_backward(dZ, cache):

    previous_A, W, b = cache
    m = previous_A.shape[1]

    dW = 1 / m * np.dot(dZ, previous_A.T)
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
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

def linear_activation_backward(dAL, cache):

    backward_cache = {}
    dA = dAL
    L = len(cache) // 2



    dZ = sigmoid_backward(dA, cache['activation_cache'+str(L)])
    dA, dW, db = linear_backward(dZ, cache['linear_cache'+str(L)])

    backward_cache['dW'+str(L)] = dW
    backward_cache['db'+str(L)] = db

    for i in reversed(range(1, L)):
        activation_cache = cache['activation_cache'+str(i)]
        linear_cache = cache['linear_cache'+str(i)]
        dZ = relu_backward(dA, activation_cache)
        dA, dW, db = linear_backward(dZ, linear_cache)

        backward_cache['dW'+str(i)] = dW
        backward_cache['db'+str(i)] = db

    return backward_cache


def update_parameters(parameters, backward_cache, learning_rate):
    L = len(parameters)// 2

    parameters = parameters.copy()


    for i in range(1, L+1):
        parameters['W' +str(i)] = parameters['W' +str(i)] - learning_rate * backward_cache['dW'+str(i)]
        parameters['b' + str(i)] = parameters['b' +str(i)] - learning_rate * backward_cache['db'+str(i)]

    return parameters


def model(train_set_picture, train_set_label, layers_size, num_iterations, learning_rate):

    costs = []
    parameters = initialize_parameters(layers_size)

    for i in range(num_iterations):

        AL, forward_cache = linear_activation_forward(train_set_picture, parameters)
        cost = compute_cost(AL, train_set_label)

        dAL = -(np.divide(train_set_label, AL) - np.divide(1 - train_set_label, 1 - AL))
        backward_cache = linear_activation_backward(dAL, forward_cache)
        parameters = update_parameters(parameters, backward_cache, learning_rate)

        if i % 100 == 0:
            if i == 0:
                print(f"Cost after {i+1} iterations: {cost}")
            else:
                print(f"Cost after {i} iterations: {cost}")

        costs.append(cost)

    return parameters, costs


def predict(X, Y, parameters):


    AL, forward_cache = linear_activation_forward(X, parameters)
    Y_prediction = np.zeros((1, Y.shape[1]))

    for i in range(Y.shape[1]):
        if AL[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0

    accuracy = str(np.sum((Y_prediction == Y) / Y.shape[1]))

    return accuracy




if __name__ == '__main__':
    train_set_picture, train_set_label, test_set_picture, test_set_label, classes = load_dataset()

    train_set_picture = reshape(train_set_picture)
    test_set_picture = reshape(test_set_picture)

    layers_dims = [12288, 35, 35, 1]
    number_iterations = 4500
    learning_rate = 0.001

    parameters, costs = model(train_set_picture, train_set_label, layers_dims, number_iterations, learning_rate)

    train_accuracy = predict(train_set_picture, train_set_label, parameters)
    test_accuracy = predict(test_set_picture, test_set_label, parameters)

    print(f"Train accuracy : {train_accuracy}")
    print(f"Test accuracy : {test_accuracy}")

    plt.plot(costs)
    plt.ylabel('Cost')
    plt.xlabel('Iterations')
    plt.title(f"Learning rate = {learning_rate}\n Layers dims = {layers_dims} Test accuracy : {test_accuracy}")
    plt.show()




