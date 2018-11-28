from neuron import Neuron
import numpy as np


class Layer:

    def __init__(self, n_neurons, n_neurons_weights):
        self.neurons = list()
        self.out_vector = np.zeros(n_neurons)
        for i in range(n_neurons):
            self.neurons.append(Neuron(n_neurons_weights))

    def compute_input_layer(self, row_input):
        for i in range(len(self.neurons)):
            self.neurons[i].compute_network_in(row_input)

    def compute_squash_layer_sigmoid(self):
        for i in range(len(self.neurons)):
            r = self.neurons[i].compute_network_out()
            self.out_vector[i] = r
        return self.out_vector


