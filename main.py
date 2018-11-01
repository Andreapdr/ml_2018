import numpy as np
from neuralNetwork import NeuralNet
import matplotlib.pyplot as plt


def get_dataset(name_csv):
    # TODO: TO BETTER IMPLEMENT
    dataset_np = np.genfromtxt(name_csv, delimiter=",")
    return dataset_np


def main():
    # Standard Monk Dataset
    train_csv = "dataset/monk1/monk1train.csv"
    test_csv = "dataset/monk1/monk1test.csv"

    # One-Hot Encoded Monk Dataset (len=17 + 1, answer at index 0)
    train_csv_one_hot = "dataset/monk2/monk2train_onehot.csv"
    test_csv_one_hot = "dataset/monk2/monk2test_onehot.csv"

    training_set = get_dataset(train_csv)
    test_set = get_dataset(test_csv)
    # training_set = get_dataset(train_csv_one_hot)
    # test_set = get_dataset(test_csv_one_hot)

    ''' NB: every layer must have as many weights as the previous layer's neuron
        SET NETWORK STRUCTURE WITH APPROPRIATE WEIGHT AMOUNTS AND LAYERS.    
         Initialize empty network = list containing layers
        set a first in layer (c neuron, d weights each)
        set out_layer (e neuron, c weights each) '''

    nn = NeuralNet()
    nn.initialize_layer(2, 6)
    nn.initialize_layer(1, 2)

    # TRAINING SESSION
    lr = 0.01
    momentum = 0.0
    nn.training(250, training_set, lr, momentum, verbose=False,
                step_decay=False, lr_decay=False)

    # TEST SESSION
    nn.test(test_set)

    # TEST PLOTTING
    # TODO: TO BETTER IMPLEMENT
    # plt.figure(figsize=(9, 8))
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


if __name__ == "__main__":
    main()
