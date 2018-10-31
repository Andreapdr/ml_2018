import numpy as np
from neuralNetwork import NeuralNet
import matplotlib.pyplot as plt


def preprocess_dataset(name_csv):
    # TODO: TO BETTER IMPLEMENT
    dataset_np = np.genfromtxt(name_csv, delimiter=",")
    return dataset_np


if __name__ == "__main__":

    train_csv = "monk_dataset.csv"
    test_csv = "monk_dataset_test.csv"

    training_set = preprocess_dataset(train_csv)
    test_set = preprocess_dataset(test_csv)
    # test_data = np.array([[1, 2, 3, 10]])       # Watch out: length of dataset and test_data is computed DIFFERENTLY

    ''' NB: every layer must have as many weights as the previous layer's neuron
        SET NETWORK STRUCTURE WITH APPROPRIATE WEIGHT AMOUNTS AND LAYERS
        NB: SET ON PRE-PROCESSING MONK DATASET in TRAINING FUNCTION '''

    nn = NeuralNet()            # initialize empty network = list containing layers
    nn.initialize_layer(3, 6)   # set a first in layer (3 neuron, 6 weights each)
    nn.initialize_layer(1, 3)   # set out_layer (1 neuron, 3 weights each)

    # TRAINING SESSION
    nn.training(450, training_set, learning_rate=0.10, momentum=0.01, verbose=False, monksDataset=True)

    # TEST SESSION
    nn.test(test_set, monksDataset=True)

    # TEST PLOTTING
    # TODO: TO BETTER IMPLEMENT
    cord_x = list()
    cord_y = list()
    for elem in nn.error_list:
        cord_x.append(elem[0])
        cord_y.append(elem[1])
    plt.plot(cord_x, cord_y, )
    plt.xlabel("Epochs number")
    plt.ylabel("Error")
    plt.show()

