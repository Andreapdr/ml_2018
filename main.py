import numpy as np
from neuralNetwork import NeuralNet


def preprocess_dataset(name_csv):
    dataset_np = np.genfromtxt(name_csv, delimiter=",")
    return dataset_np


if __name__ == "__main__":
    name_csv = "monk_dataset.csv"

    dataset= preprocess_dataset(name_csv)

    nn = NeuralNet()  # initialize empty network = list containing layers
    nn.initialize_layer(6, 6)  # set a first in layer (2 neuron, 2 weights each)
    nn.initialize_layer(3, 6)  # set hidden layer (3 neuron, 2 weights each)
    nn.initialize_layer(1, 3)  # set out_layer (1 neuron, 3 weights each)

    # print("Weights as initialized")
    # for layer in nn.layer_list:
    #     for neuron in layer.neurons:
    #         print(neuron.weights)

    nn.training(1, dataset, verbose=False)

    # print("\nWeights after first update")
    # for layer in nn.layer_list:
    #     for neuron in layer.neurons:
    #         print(neuron.weights)

# TODO: implement a way to assess error in training


