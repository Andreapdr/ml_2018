from neuralNetwork import NeuralNet, sigmoid_function, derivative_sigmoid, tanh_function, tanh_derivative
from utils import get_dataset, horror_plot, horror_plot2
import numpy as np


def main():
    train_csv_one_hot = "dataset/monk1/monk1train_onehot.csv"
    test_csv_one_hot = "dataset/monk1/monk1test_onehot.csv"
    train_csv = "dataset/monk1/monk1train.csv"
    test_csv = "dataset/monk1/monk1test.csv"
    testing_cup = "dataset/blindcup/LOC-OSM2-TR.csv"

    training_set = get_dataset(train_csv_one_hot)
    test_set = get_dataset(test_csv_one_hot)

    testing_data = np.array([[10, 1, 2, 3], [10, 1, 2, 3]])

    for i in range(1):
        nn = NeuralNet()
        nn.init_input_layer(17)
        nn.init_layer(6, 17)
        nn.init_layer(2, 6)
        nn.init_layer(1, 2)

        lr = 0.03
        epochs = 50
        activation_function = tanh_function
        derivative_activation = tanh_derivative
        nn.train(training_set, test_set, epochs, lr, activation_function, derivative_activation)

        horror_plot(nn, lr, 0)


if __name__ == "__main__":
    main()
