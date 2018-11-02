from neuralNetwork import NeuralNet
import matplotlib.pyplot as plt
from utils import get_dataset


def main():
    # Standard Monk Dataset
    train_csv = "dataset/monk3/monk3train.csv"
    test_csv = "dataset/monk3/monk3test.csv"

    # One-Hot Encoded Monk Dataset (len=17 + 1, answer at index 0)
    train_csv_one_hot = "dataset/monk3/monk3train_onehot.csv"
    test_csv_one_hot = "dataset/monk3/monk3test_onehot.csv"

    # training_set = get_dataset(train_csv)
    # test_set = get_dataset(test_csv)
    training_set = get_dataset(train_csv_one_hot)
    test_set = get_dataset(test_csv_one_hot)

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
    lr = 1.00
    momentum = 0.3
    alpha = 0.001
    nn.training(50, training_set, lr, momentum, alpha, verbose=False,
                step_decay=False, lr_decay=True)

    # TEST SESSION
    nn.test(test_set)

    # TRAINING SESSION PLOTTING
    # TODO: TO BETTER IMPLEMENT
    plt.title(f"Error Function MSE. \nlr: {lr}, momentum: {momentum}")
    cord_x = list()
    cord_y = list()
    acc_x = list()
    acc_y = list()
    for elem in nn.error_list:
        cord_x.append(elem[0])
        cord_y.append(elem[1])
    plt.plot(cord_x, cord_y, label="Error Rate")
    # for elem in nn.accuracy_list:
    #     acc_x.append(elem[0])
    #     acc_y.append(elem[1])
    # plt.plot(acc_x, acc_y, label="Accuracy")
    # plt.xlabel("Epochs number")
    # plt.ylabel("Error")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
