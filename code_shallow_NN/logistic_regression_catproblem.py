# -*- coding: utf-8 -*-


import numpy as np
import copy
import matplotlib as plt
import scipy
import h5py
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset


def sigmoid(x):
    s = 1/(1 + np.exp(-x))
    return s
    
def initialize_w_b(dim):
    w = np.zeros((dim, 1))
    b = 0.
    return w, b

def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = -1/m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    dw = 1/m * np.dot(X, (A - Y).T)
    db = 1/m * np.sum(A - Y)
    grads = {"dw":dw,
             "db":db}
    
    return grads, cost
    
def optimize(w, b, X, Y, iterations_number, learning_rate):
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    for i in range(iterations_number):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate * dw
        b = b - learning_rate * db
    params = {"w":w,
              "b":b}
    
        
    return params, grads
   
def predict(w, b, X):
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
    w, b = initialize_w_b(X_train.shape[0])
    params, grads = optimize(w, b, X_train, Y_train, iterations_number, learning_rate)
    w = params["w"]
    b = params["b"]
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    train_accuracy = 100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100
    test_accuracy = 100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100
    predictions = {"Y_prediction_test":Y_prediction_test,
                   "Y_prediction_train":Y_prediction_train}
    
    return train_accuracy, test_accuracy, predictions, params
    
    
      
            
'''    
# Load data
train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()

# Reshape/Unroll 
train_set_x = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# Standardize dataset
train_set_x = train_set_x / 255
test_set_x = test_set_x / 255

train_accuracy, test_accuracy = logistic_regression_model(train_set_x, train_set_y_orig, test_set_x, test_set_y_orig, 2000, 0.005)
'''

