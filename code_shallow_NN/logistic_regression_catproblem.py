# -*- coding: utf-8 -*-


import numpy as np
import copy
import matplotlib.pyplot as plt
import scipy
import h5py
from lr_utils import load_dataset


def sigmoid(x):
    """
    Compute the sigmoid of x

    Parameters
    ----------
    x : scalar or numpy array
        A scalar or numpy array of any size.

    Returns
    -------
    s : scalar or numpy array
        sigmoid(x).

    """   
    s = 1/(1 + np.exp(-x))
    return s
    
def initialize_w_b(dim):
    """
    Initialize w and b to zeros.

    Parameters
    ----------
    dim : scalar
          Size of the w vector we want.

    Returns
    -------
    w : numpy array
        Initialized vector of shape (dim, 1).
    b : scalar
        Initialized scalar.

    """
    w = np.zeros((dim, 1))
    b = 0.
    return w, b

def propagate(w, b, X, Y):
    """
    Implement forward propagation(Compute the cost and gradients) 

    Parameters
    ----------
    w : numpy array
        Parameters(weights) that used to implement linear part of the forward propagation.
    b : scalar
        Parameters(bias) that used to implement linear part of the forward propagation.
    X : scalar or array
        Examples.
    Y : scalar or array
        True "label" for the inputs.

    Returns
    -------
    grads : dictionary
        A dictionary that contains gradients of the loss.
    cost : narray
        negative log-likelihood cost for logistic regression.

    """
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = -1/m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    dw = 1/m * np.dot(X, (A - Y).T)
    db = 1/m * np.sum(A - Y)
    grads = {"dw":dw,
             "db":db}
    
    return grads, cost
    
def optimize(w, b, X, Y, iterations_number, learning_rate):
    """
    Optimize w and b by running a gradient descent algorithm

    Parameters
    ----------
    w : numpy array
        Weights.
    b : scalar
        Bias.
    X : scalar or array
        Examples.
    Y : scalar or array
        True "label" for the inputs.
    iterations_number : scalar
        Number of iterations of the optimization loop.
    learning_rate : scalar
        Learning rate of the gradient descent update rule.

    Returns
    -------
    params : dictionary
        A dictionary contains updated parameters(weights w and bias b).
    grads : dictionary
        A dictionary that contains gradients of the loss.

    """
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    costs = []
    for i in range(iterations_number):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate * dw
        b = b - learning_rate * db
        costs.append(cost)
    params = {"w":w,
              "b":b}
    
        
    return params, grads, costs
   
def predict(w, b, X):
    """
    

    Parameters
    ----------
    w : numpy array
        Updated weights.
    b : scalar
        Updated bias.
    X : scalar or array
        Examples.

    Returns
    -------
    Y_prediction : numpy array
        A numpy array (vector) containing all predictions (0/1) for the examples in X.

    """
    Y_prediction = np.zeros((1, X.shape[1]))
    w = w.reshape(X.shape[0],1)
    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(X.shape[1]):
        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0
    return Y_prediction 

def logistic_regression_model(X_train, Y_train, X_test, Y_test, iterations_number, learning_rate):
    """
    Build the logistic regression model by calling the function that implemented previously

    Parameters
    ----------
    X_train : numpy array
        training set represented by a numpy array of shape.
    Y_train : numpy array
        training labels represented by a numpy array (vector) of shape.
    X_test : numpy array
        test set represented by a numpy array of shape.
    Y_test : numpy array
        test labels represented by a numpy array (vector) of shape.
    iterations_number : scalar
        hyperparameter representing the number of iterations to optimize the parameters.
    learning_rate : scalar
        hyperparameter representing the learning rate used in the update rule of optimize().

    Returns
    -------
    train_accuracy : scalar
        Accuracy of the model while running on the training set.
    test_accuracy : scalar
        Accuracy of the model while running on the test set.

    """
    w, b = initialize_w_b(X_train.shape[0])
    params, grads, costs = optimize(w, b, X_train, Y_train, iterations_number, learning_rate)
    w = params["w"]
    b = params["b"]
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    train_accuracy = 100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100
    test_accuracy = 100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100
    
    return train_accuracy, test_accuracy, params, costs
    
    
      
            
'''    
# Load data from dataset
train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()

# Example of a picture
plt.imshow(train_set_x_orig[18])

# Reshape/Unroll training and test examples
train_set_x = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# Standardize dataset
train_set_x = train_set_x / 255
test_set_x = test_set_x / 255

# Make prediction and calculate accuracy respectively 
train_accuracy, test_accuracy, costs = logistic_regression_model(train_set_x, train_set_y_orig, test_set_x, test_set_y_orig, 2000, 0.005)
'''
