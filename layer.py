from neuron import Neuron
import numpy as np


class Layer:

    def __init__(self, n_neurons, n_neurons_weights):
        self.neurons = list()

        # self.out_vector = np.random.uniform(size=(1, n_neurons))
        self.out_vector = np.zeros((1, n_neurons))
        for i in range(n_neurons):
            self.neurons.append(Neuron(n_neurons_weights))

    def compute_input_layer(self, row_input):
        for i in range(len(self.neurons)):
            self.neurons[i].compute_network_in(row_input)

    def compute_squash_layer_sigmoid(self):
        for i in range(len(self.neurons)):
            r = self.neurons[i].compute_output_sigmoid()
            self.out_vector[0, i] = r
        return self.out_vector

    def compute_squash_layer_crossentropy(self):
        for i in range(len(self.neurons)):
            r = self.neurons[i].compute_output_crossentropy()
            self.out_vector[0, i] = r
        return self.out_vector

