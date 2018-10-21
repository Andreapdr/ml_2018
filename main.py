import numpy as np
from neuralNetwork import NeuralNet


tr_set = np.array([  [1,1,1,2,3,1,2],
                    [1,1,2,1,1,1,2],
                    [0,1,2,1,1,2,1],
                    [0,1,2,1,1,3,1],
                    [0,1,2,1,1,4,2],
                    [1,1,2,1,2,1,1],
                    [0,1,2,1,2,3,1],
                    [0,1,2,1,2,3,2],
                    [0,1,2,1,2,4,2],
                    [0,1,2,1,3,2,1],
                    [0,1,2,1,3,4,2]])


if __name__ == "__main__":
    nn = NeuralNet()  # initialize empty network = list containing layers
    nn.initialize_layer(6, 6)  # set a first in layer (2 neuron, 2 weights each)
    nn.initialize_layer(3, 6)  # set hidden layer (3 neuron, 2 weights each)
    nn.initialize_layer(2, 3)  # set out_layer (2 neuron, 3 weights each)

    # print("Weights as initialized")
    # for layer in nn.layer_list:
    #     for neuron in layer.neurons:
    #         print(neuron.weights)

    nn.training(100, tr_set)

    # print("\nWeights after first update")
    # for layer in nn.layer_list:
    #     for neuron in layer.neurons:
    #         print(neuron.weights)




