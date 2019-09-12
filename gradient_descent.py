from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
import numpy as np
from matplotlib import pylab as plt

import pdb;



def predict_linear_regression(X, theta):
    X = np.insert(X, 0, values=1, axis=1)
    pred = np.dot(X, theta)
    return pred


def my_normalize(X):
    # column-wise mean
    X_norm = np.mean(X, axis=0)
    X_norm = X_norm - X
    sigma = np.std(X, axis=0)
    X_norm /= sigma
    return X_norm


def compute_cost(X, y, theta):
    # computes the cost of theta as parameter for linear regression to fit the
    # data point in X and y
    no_of_rows = len(y)  # number of training examples
    J = 0  # J is the cost
    h = np.dot(X, theta)
    square_error = np.power((h - y), 2)
    J = (1.0 / (2.0 * no_of_rows)) * np.sum(square_error)

    return J


def gradient_descent(X, Y, alpha=1e-7, nepochs=100):
    # insert a column at 0th element
    X = normalize(X)
    X = np.insert(X, 0, values=1, axis=1)
    theta = np.zeros((X.shape[1], 1))
    y = np.asmatrix(Y)
    y = y.T

  #  pdb.set_trace()
    history = []
    alpha /= len(y)
    for _ in range(nepochs):
        error = np.dot(X, theta) - y
        gradient = np.dot(error.T, X).T
        theta -= alpha * gradient
        #history.append(mean_squared_error(np.dot(X, theta), y))
        history.append(compute_cost(X,y,theta))
    return theta, history


ds = datasets.load_boston()
X = ds.data
Y = ds.target
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.4, random_state=42)
gradient_descent_theta, history = gradient_descent(train_X, train_Y, alpha=1e-6, nepochs=100)
predicted_Y = predict_linear_regression(test_X, gradient_descent_theta)
print(mean_squared_error(predicted_Y, test_Y))
plt.plot(history)
plt.show()

#pdb.set_trace()
