import numpy as np


class Layer:

    def __init__(self, n_neurons, n_weights):
        self.neurons = np.ones(n_neurons)
        self.weights = np.random.uniform(low=-0.70, high=0.70, size=[n_neurons, n_weights])
        self.delta = np.zeros(n_neurons)



