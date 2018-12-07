import numpy as np

# TODO: should differentiate between input (net) and output (out)


class Layer:

    def __init__(self, n_neurons, n_weights, activation):
        self.neurons = np.ones(n_neurons)
        self.out = np.ones(n_neurons)
        self.weights = np.random.uniform(low=-0.70, high=0.70, size=[n_neurons, n_weights])
        self.delta = np.zeros(n_neurons)
        self.bias = np.random.uniform(low=-0.50, high=0.50, size=[n_neurons])
        self.previous_update = np.random.uniform(low=-0.70, high=0.70, size=[n_neurons, n_weights])
        self.previous_bias_update = np.random.uniform(low=-0.50, high=0.50, size=[n_neurons])
        self.activation_function = activation

    def activation_layer(self):
        if self.activation_function == "sigmoid":
            self.neurons = sigmoid_function(self.neurons)
            # self.out = sigmoid_function(self.neurons)
        elif self.activation_function == "tanh":
            self.neurons = tanh_function(self.neurons)
            # self.out = tanh_function(self.neurons)
        elif self.activation_function == "linear":
            self.out = linear_activation(self.neurons)
        else:
            self.out = sigmoid_function(self.neurons)

    def activation_function_derivative(self):
        if self.activation_function == "sigmoid":
            return derivative_sigmoid(self.neurons)
            # return derivative_sigmoid(self.out)
        elif self.activation_function == "tanh":
            return tanh_derivative(self.neurons)
            # return tanh_derivative(self.out)
        elif self.activation_function == "linear":
            return linear_derivative(self.out)
        else:
            return derivative_sigmoid(self.out)


def derivative_sigmoid(x):
    return x * (1 - x)


def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))


def tanh_function(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - x**2


def linear_activation(x):
    return x


def linear_derivative(x):
    return 1
