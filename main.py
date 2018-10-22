import numpy as np
from neuralNetwork import NeuralNet


def preprocess_dataset(name_csv):
    dataset_np = np.genfromtxt(name_csv, delimiter=",")
    return dataset_np


if __name__ == "__main__":
    name_csv = "monk_dataset.csv"

    dataset = preprocess_dataset(name_csv)
    test_data = np.array([[1, 2, 3, 10]])
    # Watch out: length of dataset and test_data is computed DIFFERENTLY
    # print(len(dataset))
    # print(len(test_data))
    print(dataset.shape[0])
    print(test_data.shape[0])

    nn = NeuralNet()            # initialize empty network = list containing layers
    nn.initialize_layer(3, 6)   # set a first in layer (3 neuron, 6 weights each)
    nn.initialize_layer(3, 3)   # set hidden layer (3 neuron, 3 weights each)
    nn.initialize_layer(1, 3)   # set out_layer (1 neuron, 3 weights each)
                                # NB: every layer must have as many weights as the previous layer's neuron
                                # NB: SET NETWORK STRUCTURE WITH APPROPRIATE WEIGHT AMOUNTS AND LAYERS
                                # NB: SET ON PREPROCESSING MONK DATASET in TRAINING FUNCTION

    # Test Purpose network
    # nn = NeuralNet()
    # nn.initialize_layer(3, 3)
    # nn.initialize_layer(3, 3)
    # nn.initialize_layer(1, 3)

    # print("Weights as initialized")
    # for layer in nn.layer_list:
    #     for neuron in layer.neurons:
    #         print(neuron.weights)

    nn.training(10, dataset, learning_rate=0.002, verbose=False, monksDataset=True)

    # print("\nWeights after first update")
    # for layer in nn.layer_list:
    #     for neuron in layer.neurons:
    #         print(neuron.weights)

# TODO: implement a way to assess error in training
# TODO: INCLUDE BIAS IN ALGORITHM/WHOLE CODE

