import numpy as np


class Layer:

    def __init__(self, n_neurons, n_weights, activation):
        self.net = np.ones(n_neurons)
        self.out = np.ones(n_neurons)
        self.weights = np.random.uniform(low=-0.70, high=0.70, size=[n_neurons, n_weights])
        self.bias_W = np.random.uniform(low=-0.20, high=0.20, size=[n_neurons])
        self.delta = np.zeros(n_neurons)
        self.activation = None
        self.set_activation(activation)
        self.last_deltaW = 0

    """TESTING PURPOSES"""
    def set_weights(self, weights):
        if self.weights.shape == weights.shape:
            self.weights = weights
        else:
            print("Error: invalid W shape")

    """ According to Glorot-Bengio desired_var = uniform range in +/- 2/(n_in + n_out)
        also testing He, Rang, Zhen and Sun var = 2/n_in"""
    def set_weights_xavier(self, desired_var):
        self.weights = np.random.uniform(low=-desired_var, high=desired_var, size=np.shape(self.weights))
        self.bias_W = np.random.uniform(low=-desired_var, high=desired_var, size=np.shape(self.bias_W))

    def set_activation(self, activation):
        if activation == "sigmoid":
            self.activation = sigmoid_function
        elif activation == "linear":
            self.activation = linear_function
        elif activation == "tanh":
            self.activation = tanh_function
        elif activation == "relu":
            self.activation = relu_function

    def activation_function(self, derivative=False):
        return self.activation(self.net, derivative)


def sigmoid_function(x, derivative=False):
    if derivative:
        return sigmoid_function(x) * (1 - sigmoid_function(x))
    else:
        return 1/(1 + np.exp(-x))


def tanh_function(x, derivative=False):
    if derivative:
        return 1 - tanh_function(x) ** 2
    else:
        return np.tanh(x)


def relu_function(x, derivative=False):
    if derivative:
        return 1 / (1 + np.exp(-x))
    else:
        y = x.copy()
        return y * (y > 0)


def linear_function(x, derivative):
    if derivative:
        return 1
    else:
        return x
