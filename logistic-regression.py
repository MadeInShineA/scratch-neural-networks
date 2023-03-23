import numpy as np
from utils import *
def sigmoid(z):
    return 1/(1+np.exp(-z))


def initialize_with_zeros(size):
    w = np.zeros((size, 1))
    b = 0.0

    return w, b


def propagate(w, b, X, Y):
    m = X.shape[1]

    # Forward propagation
    A = sigmoid(np.dot(w.T, X) + b)
    cost = (-1/m) * np.sum(Y * np.log(A) + (1-Y) * np.log(1-A))

    # Backward propagation
    dw = 1/m * np.dot(X, (A-Y).T)
    db = 1/m * np.sum(A-Y)


    return dw, db, cost


def optimize(w, b, X, Y, num_iter, learning_rate):

    for i in range(num_iter):
        dw, db, cost = propagate(w, b, X, Y)

        w = w-learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            print(f"Cost for the {i} iteration : {cost}")

    return w, b, dw, db, cost


def predict (w, b, X):

    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T,X) + b)

    for i in range(A.shape[1]):
        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0

    return Y_prediction


def model (train_images, train_labels, test_images, test_labels, num_iter, learning_rate):

    w, b = initialize_with_zeros(train_images.shape[0])

    w, b, dw, db, cost = optimize(w, b, train_images, train_labels, num_iter, learning_rate)

    Y_prediction_test = predict(w, b, test_images)
    Y_prediction_train = predict(w, b, train_images)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - train_labels)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - test_labels)) * 100))


if __name__ == '__main__':
    train_set_picture, train_set_label, test_set_picture, test_set_label, classes = load_dataset()

    train_set_picture = reshape(train_set_picture)

    test_set_picture = reshape(test_set_picture)

    model(train_set_picture, train_set_label, test_set_picture, test_set_label, 2000, 0.005)







