import numpy as np
import math


class Neuron:

    def __init__(self, n_weights):
        self.weights = np.random.uniform(low=-0.70, high=0.70, size=n_weights)
        self.bias = 1.00
        self.inputs_list = np.ones(n_weights)
        self.network_in = 0.00
        self.output = 0.00
        self.delta = 0.00
        self.update_coeff = np.zeros(n_weights)

    def compute_network_in(self, row_inputs):
        for i in range(len(row_inputs)):
            input_x = row_inputs[i]
            self.inputs_list[i] = input_x
        self.network_in = np.dot(row_inputs, self.weights.T)
        self.network_in = self.network_in + self.bias
        return self.network_in

    def compute_output_sigmoid(self):
        self.output = sigmoid_function(self.network_in)
        return self.output

    # self.network in should be only positive values
    def compute_output_crossentropy(self):
        self.output = cross_entropy_function(self.network_in)
        return self.output

    def compute_output_final(self):
        out = self.output
        if out > 0.5:
            return 1
        else:
            return 0

    # for sigmoid activation function
    def compute_delta_output(self, target):
        derivative_activation_function = self.output * (1 - self.output)
        error = target - self.output
        # (target - self.output) is the derivative of the L2 loss function
        self.delta = (target - self.output) * derivative_activation_function
        return error

    def compute_delta_hidden(self, next_layer, index_of_neuron_prev_layer):
        derivative_activation_function = self.output * (1 - self.output)
        hidden_error = 0.00
        for neuron in next_layer.neurons:
            hidden_error += neuron.delta * neuron.weights[index_of_neuron_prev_layer]
        delta_hidden = hidden_error * derivative_activation_function
        self.delta = delta_hidden
        return delta_hidden


def sigmoid_function(x):
    return 1 / (1 + math.exp(-x))


def cross_entropy_function(x):
    return - np.log(x)

# test merging
