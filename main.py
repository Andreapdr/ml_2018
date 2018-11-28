from neuralNetwork import NeuralNet, sigmoid_function, derivative_sigmoid, tanh_function, tanh_derivative
from utils import get_dataset, horror_plot


def main():
    train_csv_one_hot = "dataset/monk3/monk3train_onehot.csv"
    test_csv_one_hot = "dataset/monk3/monk3test_onehot.csv"
    train_csv = "dataset/monk1/monk1train.csv"
    test_csv = "dataset/monk1/monk1test.csv"

    training_set = get_dataset(train_csv_one_hot)
    test_set = get_dataset(test_csv_one_hot)

    nn = NeuralNet()
    nn.init_inputLayer(17)
    nn.init_layer(6, 17)
    # TODO: check for multilayer problems...
    # nn.init_layer(6, 6)
    nn.init_layer(1, 6)

    # neurons = nn.get_number_neurons()
    # print(neurons)
    # weights = nn.get_number_weights()
    # print(weights)
    lr = 0.25
    epochs = 150
    nn.train(training_set, test_set, epochs, lr, sigmoid_function, derivative_sigmoid)
    horror_plot(nn, lr, 0)


if __name__ == "__main__":
    main()
