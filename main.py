from neuralNetwork import NeuralNet
import matplotlib.pyplot as plt
from utils import get_dataset


def main():
    # Standard Monk Dataset
    train_csv = "dataset/monk1/monk1train.csv"
    test_csv = "dataset/monk1/monk1test.csv"

    # One-Hot Encoded Monk Dataset (len = 16 + 1, answer at index 0)
    train_csv_one_hot = "dataset/monk1/monk1train_onehot.csv"
    test_csv_one_hot = "dataset/monk1/monk1test_onehot.csv"

    training_set = get_dataset(train_csv_one_hot)
    test_set = get_dataset(test_csv_one_hot)
    # training_set = get_dataset(train_csv)
    # test_set = get_dataset(test_csv)

    ''' NB: every layer must have as many weights as the previous layer's neuron
        SET NETWORK STRUCTURE WITH APPROPRIATE WEIGHT AMOUNTS AND LAYERS.    
         Initialize empty network = list containing layers
        set a first in layer (c neuron, d weights each)
        set out_layer (e neuron, c weights each) '''

    # INITIALIZATION
    nn = NeuralNet()
    nn.initialize_layer(3, 17)
    nn.initialize_layer(1, 3)

    # TRAINING SESSION
    lr = 0.03
    momentum = 0.0
    # TODO: Check alpha not working as intended
    alpha = 0.00
    nn.training(250, training_set, test_set, lr, momentum, alpha, verbose=False,
                step_decay=True, lr_decay=False)

    # TEST SESSION
    nn.test(test_set)

    # TRAINING SESSION PLOTTING
    # TODO: TO BETTER IMPLEMENT
    plt.title(f"Error Function MSE \nlr: {lr}, momentum: {momentum}")
    cord_x = list()
    cord_y = list()
    cord_x_test = list()
    cord_y_test = list()
    for elem in nn.error_list:
        cord_x.append(elem[0])
        cord_y.append(elem[1])
    for elem in nn.error_list_test:
        cord_x_test.append(elem[0])
        cord_y_test.append(elem[1])
    plt.plot(cord_x, cord_y, label="Error Rate Training")
    plt.plot(cord_x_test, cord_y_test, label="Error Rate Validation")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()

# test merging
