import numpy as np
from neuralNetwork import NeuralNet
import matplotlib.pyplot as plt


def preprocess_dataset(name_csv):
    # TODO: TO BETTER IMPLEMENT
    dataset_np = np.genfromtxt(name_csv, delimiter=",")
    return dataset_np


if __name__ == "__main__":

    train_csv = "dataset/monk1/monk1train.csv"
    test_csv = "dataset/monk1/monk1test.csv"

    training_set = preprocess_dataset(train_csv)
    test_set = preprocess_dataset(test_csv)

    ''' NB: every layer must have as many weights as the previous layer's neuron
        SET NETWORK STRUCTURE WITH APPROPRIATE WEIGHT AMOUNTS AND LAYERS
        NB: SET ON PRE-PROCESSING MONK DATASET in TRAINING FUNCTION '''

    nn = NeuralNet()            # initialize empty network = list containing layers
    nn.initialize_layer(3, 6)   # set a first in layer (3 neuron, 6 weights each)
    nn.initialize_layer(1, 3)   # set out_layer (1 neuron, 3 weights each)

    # TRAINING SESSION
    lr = 0.10
    momentum = 0.50
    nn.training(500, training_set, lr, momentum, verbose=False, monksDataset=False,
                step_decay=False, lr_decay=False)

    # TEST SESSION
    nn.test(test_set, monksDataset=False)

    # TEST PLOTTING
    # TODO: TO BETTER IMPLEMENT
    plt.title(f"Loss Function plot. \nlr: {lr}, momentum: {momentum}")
    cord_x = list()
    cord_y = list()
    for elem in nn.error_list:
        cord_x.append(elem[0])
        cord_y.append(elem[1])
    plt.plot(cord_x, cord_y)
    plt.xlabel("Epochs number")
    plt.ylabel("Error")
    plt.show()

