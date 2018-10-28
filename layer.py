from neuron import Neuron
import numpy as np


class Layer:

    def __init__(self, n_neurons, n_neurons_weights):
        self.neurons = list()
        self.out_vector = np.ones([1, n_neurons])
        for i in range(n_neurons):
            self.neurons.append(Neuron(n_neurons_weights))

    def compute_input_layer(self, row_input):
        for i in range(len(self.neurons)):
            self.neurons[i].compute_network_in(row_input)

    def compute_squash_layer(self):
        for i in range(len(self.neurons)):
            r = self.neurons[i].compute_output()
            self.out_vector[0, i] = r
        return self.out_vector

    def compute_squash_layer_final(self):
        for i in range(len(self.neurons)):
            r = self.neurons[i].compute_output_final()
            self.out_vector[0, i] = r
        return  self.out_vector
