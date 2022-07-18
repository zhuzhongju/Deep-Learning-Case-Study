# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
import logistic_regression_catproblem as lr


def layer_size(X, Y):
    """
    Define model sturcture

    Parameters
    ----------
    X : numpy array
        Examples.
    Y : numpy array
        Ture label for examples.

    Returns
    -------
    n_x : scalar
        Size of the input layer.
    n_h : scalar
        Size of the hidden layer.
    n_y : scalar
        Size of the output layer.

    """
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    return n_x, n_h, n_y


def initialize_parameters(n_x, n_h, n_y):
    """
    Initialize weights w and bias b randomly.

    Parameters
    ----------
    n_x : scalar
        Size of the input layer.
    n_h : scalar
        Size of the hidden layer.
    n_y : scalar
        Size of the output layer.

    Returns
    -------
    parameters : dictionary
        A dictionary that contians initialized weights and bias.

    """
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
    """
    Implement forward propagation

    Parameters
    ----------
    parameters : dictionary
        A dictionary that contains weights and bias.
    X : numpy array
        Examples

    Returns
    -------
    cache : dictionary
        A dictionary that contains intermediate values.

    """
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
    """
    Compute cost

    Parameters
    ----------
    cache : dictionary
        Intermediate values that were calculated from forward propagation.
    Y : numpy array
        True label for examples.

    Returns
    -------
    cost : numpy array
        negative log-likelihood cost for neural networks.

    """
    A2 = cache["A2"]
    m = Y.shape[1]
    cost = -1/m * np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))
    cost = float(np.squeeze(cost))
    return cost

def backward(parameters, cache, X, Y):
    """
    Implement backward propagation.

    Parameters
    ----------
    parameters : dictionary
        A dictionary that contains weights and bias.
    cache : dictionary
        Intermediate values that were calculated from forward propagation.
    X : numpy array
        Examples.
    Y : numpy array
        Ture label for examples.

    Returns
    -------
    grads : dictionary
        A dictionary that contains gradients of the loss.

    """
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
    """
    Update parameters after every iteraton.

    Parameters
    ----------
    grads : dictionary
        A dictionary that contains gradients of the loss.
    parameters : dictionary
        A dictionary that contains weights and bias.
    learning_rate : scalar
        Learning rate of the optimization algorithm.

    Returns
    -------
    parameters : dictionary
        A dictionary contains updated parameters(weights w and bias b).

    """
    parameters["W2"] = parameters["W2"] - learning_rate * grads["dW2"]
    parameters["b2"] = parameters["b2"] - learning_rate * grads["db2"]
    parameters["W1"] = parameters["W1"] - learning_rate * grads["dW1"]
    parameters["b1"] = parameters["b1"] - learning_rate * grads["db1"]
    return parameters

def predict(parameters, X):
    """
    Make prediction based on the well-trained parameters.

    Parameters
    ----------
    parameters : dictionary
        A dictionary that contains weights and bias.
    X : numpy array
        Examples.

    Returns
    -------
    Y_predictions : numpy array
        A numpy array (vector) containing all predictions (0/1) for the examples in X.

    """
    cache = forward(parameters, X)
    Y_predictions = (cache["A2"] > 0.5)
    return Y_predictions
 
def nn_model(X_train, Y_train, learning_rate, iterations_num):
    """
    Put functions together to build a 2-layers NN model

    Parameters
    ----------
    X_train : numpy array
        training set represented by a numpy array of shape.
    Y_train : numpy array
        training labels represented by a numpy array (vector) of shape.
    learning_rate : scalar
        hyperparameter representing the learning rate used in the update rule of optimize().
    learning_rate : scalar
        hyperparameter representing the learning rate used in the update rule of optimize().

    Returns
    -------
    parameters : dictionary
        A dictionary that contains well-trained parameters(weights w and bias b).
    costs : list
        A list contains negative log-likelihood cost for neural networks for every iteration.

    """
    costs = []
    n_x, n_h, n_y = layer_size(X_train, Y_train)
    parameters = initialize_parameters(n_x, n_h, n_y)
    for i in range(iterations_num):
        cache = forward(parameters, X_train)
        cost = compute_cost(cache, Y_train)
        grads = backward(parameters, cache, X_train, Y_train)
        parameters = update_params(grads, parameters, learning_rate) 
        costs.append(cost)
    return parameters, costs
    
# load data by using pre-built function and visualize them   
X, Y = load_planar_dataset()
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);

# Train a 2-layer NN model and make a prediction based on it.
parameters, costs = nn_model(X, Y, learning_rate = 1.2, iterations_num = 1000)
predictions = predict(parameters, X)

# Utilize logistic regression model to predict.
train_accuracy, test_accracy, LR_params, LR_costs = lr.logistic_regression_model(X, Y, X, Y, iterations_number = 1000, learning_rate = 1.2)   

# Visualize outcomes
plt.figure(2) 
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)

plt.figure(3)
plt.title("NN_cost")
plt.xlabel("Number of Iterations")
plt.ylabel("cost")
plt.plot(np.linspace(0, len(costs), len(costs)), costs)

plt.figure(4) 
plot_decision_boundary(lambda x: lr.predict(LR_params["w"], LR_params["b"], x.T), X, Y)

print ('2-layer nn model_accuracy: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')    
print ('Logistic_regression accuracy: %d' % test_accracy + '%')
    
    
    
       
    
    
    
    
