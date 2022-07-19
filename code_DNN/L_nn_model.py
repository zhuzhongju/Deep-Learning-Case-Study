# -*- coding: utf-8 -*-
import numpy as np
import time
import h5py
import scipy
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward, load_data



def initialize_parameters(layer_dims):
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(layer_dims[l-1]) #* 0.01
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))       
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))             
    return parameters
        
def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache        

def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    cache = (linear_cache, activation_cache)
    return A, cache
    

def L_model_forward(parameters, X):
    caches = []
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)] , parameters["b" + str(l)] , "relu")
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters["W" + str(L)] , parameters["b" + str(L)] , "sigmoid")
    caches.append(cache)
    return AL, caches

def compute_cost(Y, AL):
    m = Y.shape[1]
    cost = -1/m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
    cost = np.squeeze(cost)
    return cost

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    L = len(caches)
    grads = {}
    Y = Y.reshape(AL.shape)
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        grads["dA" + str(l)], grads["dW" + str(l + 1)], grads["db" + str(l + 1)] = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, "relu")
    return grads

def update_params(parameters, grads, learning_rate):
    parameters = parameters.copy()
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)] 
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    return parameters

def L_nn_model(layer_dims, X, Y, iterations_num, learning_rate):
    costs = []
    parameters = initialize_parameters(layer_dims)
    for i in range(0, iterations_num):
        AL, caches = L_model_forward(parameters, X) 
        cost = compute_cost(Y, AL)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_params(parameters, grads, learning_rate)
        costs.append(cost)
    return parameters, costs

def predict(parameters, X):
    AL, caches = L_model_forward(parameters, X)
    predictions = (AL > 0.5)
    return predictions

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.
n_x = 12288     # num_px * num_px * 3
n_h = 7
n_y = 1
#layer_dims = (n_x, n_h, n_y)
layer_dims = (12288, 20, 7, 5, 1)
parameters, cost = L_nn_model(layer_dims, train_x, train_y, iterations_num = 3000, learning_rate = 0.0075)
Y_hat = predict(parameters, test_x)
print ('Accuracy: %d' % float((np.dot(test_y, Y_hat.T) + np.dot(1 - test_y, 1 - Y_hat.T)) / float(test_y.size) * 100) + '%')    
