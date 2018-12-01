from neuralNetwork import NeuralNet, sigmoid_function, derivative_sigmoid, tanh_function, tanh_derivative
from utils import get_dataset, horror_plot
import multiprocessing as mp


def main():
    train_csv_one_hot = "dataset/monk1/monk1train_onehot.csv"
    test_csv_one_hot = "dataset/monk1/monk1test_onehot.csv"
    train_csv = "dataset/monk1/monk1train.csv"
    test_csv = "dataset/monk1/monk1test.csv"

    training_set = get_dataset(train_csv_one_hot)
    test_set = get_dataset(test_csv_one_hot)

    nn = NeuralNet()
    nn.init_inputLayer(17)
    nn.init_layer(4, 17)
    # TODO: check for multilayer problems + WEIRD PROBLEM FOR LOW LEARNING RATE :(
    # nn.init_layer(6, 6)
    nn.init_layer(1, 4)

    # neurons = nn.get_number_neurons()
    # print(neurons)
    # weights = nn.get_number_weights()
    # print(weights)

    lr = 0.30
    epochs = 50
    activation_function = sigmoid_function
    derivative_activation = derivative_sigmoid
    nn.train(training_set, test_set, epochs, lr, activation_function, derivative_activation)
    horror_plot(nn, lr, 0)


if __name__ == "__main__":
    main()
