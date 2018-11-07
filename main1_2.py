import numpy as np
import matplotlib.pyplot as plt
from neuralNetwork1_2 import NeuralNet


def preprocess_dataset(name_csv):
    dataset_np = np.genfromtxt(name_csv, delimiter=",")
    return dataset_np


def show_plot(functions):
    for fun in functions:
        elem_x, elem_y = list(), list()
        for elem in fun:
            elem_x.append(elem[0])
            elem_y.append(elem[1])
        plt.plot(elem_x, elem_y)
    plt.xlabel("Epochs number")
    plt.ylabel("Error")
    plt.show()


if __name__ == "__main__":
    tr_set = preprocess_dataset("monk_dataset.csv")
    val_set = preprocess_dataset("monk_dataset_test.csv")

    # test_data = np.array([[1, 2, 3, 10]])  # Watch out: length of dataset and test_data is computed DIFFERENTLY

    ''' NB: every layer must have as many weights as the previous layer's neuron
        SET NETWORK STRUCTURE WITH APPROPRIATE WEIGHT AMOUNTS AND LAYERS
        NB: SET ON PRE-PROCESSING MONK DATASET in TRAINING FUNCTION '''

    nn = NeuralNet()  # initialize empty network = list containing layers

    nn.initialize_layer(10, 6)  # set a first in layer (3 neuron, 6 weights each)
    nn.initialize_layer(10, 10)  # set a first in layer (3 neuron, 6 weights each)
    nn.initialize_layer(1, 10)  # set out_layer (1 neuron, 3 weights each)

    nn.training(300, tr_set, val_set, learning_rate=0.01, momentum=0.01, verbose=False, monksDataset=True)
    nn.test(val_set, monksDataset=True, verbose=True)

    error_lists = list()
    error_lists.append(nn.tr_error_list)
    error_lists.append(nn.val_error_list)
    show_plot(error_lists)

    # Test Purpose network
    # nn = NeuralNet()
    # nn.initialize_layer(150, 6)
    # nn.initialize_layer(150, 150)
    # nn.initialize_layer(1, 150)

    # nn.training(1000, test_data, learning_rate=0.5, monksDataset=False)
    # nn.training(1000, dataset, learning_rate=0.5, verbose=False, monksDataset=True)
# TODO: implement a way to assess error in training
# TODO: INCLUDE BIAS IN ALGORITHM/WHOLE CODE
