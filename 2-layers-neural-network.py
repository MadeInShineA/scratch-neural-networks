import numpy as np
import matplotlib.pyplot as plt
from utils import *
def initialize_parameters(x_size,hidden_size,output_size):

    W1 = np.random.randn(hidden_size, x_size) * 0.01
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(output_size, hidden_size) * 0.01
    b2 = np.zeros((output_size, 1))

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

def compute_cost(A2, Y):

    m = Y.shape[1]
    cost = -1/m * np.sum(Y * np.log(A2) + (1-Y) * np.log(1-A2))

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

    if activation == "sigmoid":
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


def predict (W1, b1, W2, b2, X, Y):

    A1, cache1 = linear_activation_forward(X, W1, b1, "relu")
    A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")

    Y_prediction = np.zeros((1, Y.shape[1]))

    for i in range(Y.shape[1]):
        if A2[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0

    accuracy = str(np.sum((Y_prediction == Y) / Y.shape[1]))
    return accuracy


def model(train_set_picture, train_set_label, layers_dims, num_itterations, learning_rate):

    costs = []
    x_size, hidden_size, output_size = layers_dims
    W1, b1, W2, b2 = initialize_parameters(x_size, hidden_size, output_size)

    for i in range(0,num_itterations):
        A1, cache1 = linear_activation_forward(train_set_picture,W1,b1,"relu")
        A2,cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")
        cost = compute_cost(A2, train_set_label)

        # Noter !
        dA2 = -(np.divide(train_set_label, A2) - np.divide(1-train_set_label, 1-A2))
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")

        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)

        if i % 100 == 0:
            print(f"Cost after {i} itterations : {cost}")
        costs.append(cost)

    return W1, b1, W2, b2, costs


if __name__ == '__main__':
    train_set_picture, train_set_label, test_set_picture, test_set_label, classes = load_dataset()

    train_set_picture = reshape(train_set_picture)
    test_set_picture = reshape(test_set_picture)

    layers_dims = [train_set_picture.shape[0], 7, 1]
    learning_rate = 0.005

    W1, b1, W2, b2, costs = model(train_set_picture, train_set_label, layers_dims, 2000, learning_rate)
    train_accuracy = predict(W1, b1, W2, b2, train_set_picture, train_set_label)
    test_accuracy = predict(W1, b1, W2, b2, test_set_picture, test_set_label)

    plt.plot(costs)
    plt.ylabel('Cost')
    plt.xlabel('Iterations')
    plt.title(f"Learning rate = {learning_rate}\n Layers dims = {layers_dims} Test accuracy : {test_accuracy}")
    plt.show()






