# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
import logistic_regression_catproblem as lr


def layer_size(X, Y):
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    return n_x, n_h, n_y


def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters
    
def forward(parameters, X):
    Z1 = np.dot(parameters["W1"], X) + parameters["b1"]
    A1 = np.tanh(Z1)
    Z2 = np.dot(parameters["W2"], A1) + parameters["b2"]
    A2 = sigmoid(Z2)
    cache = {"Z1":Z1,
             "A1":A1,
             "Z2":Z2,
             "A2":A2}
    return cache

def compute_cost(cache, Y):
    A2 = cache["A2"]
    m = Y.shape[1]
    cost = -1/m * np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))
    cost = float(np.squeeze(cost))
    return cost

def backward(parameters, cache, X, Y):
    m = Y.shape[1]
    A2 = cache["A2"]
    dZ2 = A2 - Y
    dW2 = 1/m * np.dot(dZ2, cache["A1"].T)
    db2 = 1/m * np.sum(dZ2, axis = 1, keepdims = True)
    dZ1 = np.dot(parameters["W2"].T, dZ2) * (1 - np.power(cache["A1"], 2))
    dW1 = 1/m * np.dot(dZ1, X.T)
    db1 = 1/m * np.sum(dZ1, axis = 1, keepdims = True)
    grads = {"dW2":dW2,
             "db2":db2,
             "dW1":dW1,
             "db1":db1}
    return grads

def update_params(grads, parameters, learning_rate):
    parameters["W2"] = parameters["W2"] - learning_rate * grads["dW2"]
    parameters["b2"] = parameters["b2"] - learning_rate * grads["db2"]
    parameters["W1"] = parameters["W1"] - learning_rate * grads["dW1"]
    parameters["b1"] = parameters["b1"] - learning_rate * grads["db1"]
    return parameters

def predict(parameters, X):
    cache = forward(parameters, X)
    Y_predictions = (cache["A2"] > 0.5)
    return Y_predictions
 
def nn_model(X_train, Y_train, learning_rate, iterations_num):
    n_x, n_h, n_y = layer_size(X_train, Y_train)
    parameters = initialize_parameters(n_x, n_h, n_y)
    for i in range(iterations_num):
        cache = forward(parameters, X_train)
        cost = compute_cost(cache, Y_train)
        grads = backward(parameters, cache, X_train, Y_train)
        parameters = update_params(grads, parameters, learning_rate) 
    return parameters
    
   
X, Y = load_planar_dataset()
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);

parameters = nn_model(X, Y, learning_rate = 1.2, iterations_num = 1000)
predictions = predict(parameters, X)
plt.figure(2) 
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)

train_accuracy, test_accracy, LR_preditions, LR_params = lr.logistic_regression_model(X, Y, X, Y, iterations_number = 1000, learning_rate = 1.2)   
plt.figure(3) 
plot_decision_boundary(lambda x: lr.predict(LR_params["w"], LR_params["b"], x.T), X, Y)

print ('Accuracy: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')    
print ('Logistic_regression accuracy: %d' % test_accracy + '%')
    
    
    
       
    
    
    
    