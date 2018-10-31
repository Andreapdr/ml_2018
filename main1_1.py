import numpy as np
from neuralNetwork1_1 import NeuralNet
import matplotlib.pyplot as plt


def preprocess_dataset(name_csv):
    dataset_np = np.genfromtxt(name_csv, delimiter=",")
    return dataset_np


if __name__ == "__main__":

    name_csv = "monk_dataset.csv"

    dataset = preprocess_dataset(name_csv)
    np.random.shuffle(dataset)                  # Shuffle dataset to prevent stagnation on input patterns
    test_data = np.array([[1, 2, 3, 10]])       # Watch out: length of dataset and test_data is computed DIFFERENTLY

    ''' NB: every layer must have as many weights as the previous layer's neuron
        SET NETWORK STRUCTURE WITH APPROPRIATE WEIGHT AMOUNTS AND LAYERS
        NB: SET ON PRE-PROCESSING MONK DATASET in TRAINING FUNCTION '''

    nn = NeuralNet()            # initialize empty network = list containing layers
    nn.initialize_layer(3, 6)   # set a first in layer (3 neuron, 6 weights each)
    nn.initialize_layer(1, 3)   # set out_layer (1 neuron, 3 weights each)
    nn.training(800, dataset, learning_rate=0.02, momentum=0.5, verbose=False, monksDataset=True)
    nn.test(preprocess_dataset("monk_dataset_test.csv"), monksDataset=True)

    # TEST PLOTTING
    cord_x = list()
    cord_y = list()
    for elem in nn.error_list:
        cord_x.append(elem[0])
        cord_y.append(elem[1])
    plt.plot(cord_x, cord_y, )
    plt.xlabel("Epochs number")
    plt.ylabel("Error")
    plt.show()

    # Test Purpose network
    # nn = NeuralNet()
    # nn.initialize_layer(150, 6)
    # nn.initialize_layer(150, 150)
    # nn.initialize_layer(1, 150)

    # nn.training(1000, test_data, learning_rate=0.5, monksDataset=False)
    # nn.training(1000, dataset, learning_rate=0.5, verbose=False, monksDataset=True)
# TODO: implement a way to assess error in training
# TODO: INCLUDE BIAS IN ALGORITHM/WHOLE CODE

