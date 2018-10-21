import numpy as np
import math


class Neuron:

    def __init__(self, n_weights):
        self.weights = np.random.rand(1, n_weights)
        self.network_in = 0.00
        self.output = 0.00
        self.delta = 0.00

    def compute_network_in(self, row_inputs):
        self.network_in = np.dot(row_inputs, self.weights.T)                    # return an ndarray object !
        return self.network_in

    def get_weight(self):
        return self.weights()

    def compute_output(self):
        self.output = sigmoid_function(self.network_in)
        return self.output

    def compute_delta_output(self, target):
        derivative_activation_function = self.output * (1 - self.output)          # for sigmoidal activation function
        self.delta = (target - self.output) * derivative_activation_function
        return self.delta

    def compute_delta_hidden(self, next_layer, index_of_nueron_prev_layer):
        derivative_activation_function = self.output * (1 - self.output)          # for sigmoidal activation function
        hidden_error = 0.00
        for neuron in next_layer.neurons:                                                   # hidden_error = Delta next_layer neuron
            hidden_error += neuron.delta * neuron.weights[0, index_of_nueron_prev_layer]    # multplied by weight[j] next_layer neuron
        # print(f"For Neuron: hidden error: {hidden_error}")
        delta_hidden = hidden_error * derivative_activation_function
        # print(delta_hidden)
        self.delta = delta_hidden
        return delta_hidden


def sigmoid_function(x):
    return 1 / (1 + math.exp(-x))
