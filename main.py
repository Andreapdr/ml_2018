from neuralNetwork import NeuralNet
from utils import get_dataset, horror_plot, horror_plot2, k_fold

""" lr = Learning Rate,
    alpha = Momentum
    step_decay = value multiplying learning rate every 20 epochs - should be lower than 1 """


def test_monk():
    nn = NeuralNet("mean_squared_error")
    nn.init_input_layer(17)
    nn.init_layer(3, 17, "tanh")
    nn.init_layer(1, 3, "tanh")

    train_csv_one_hot = "dataset/monk2/monk2train_onehot.csv"
    test_csv_one_hot = "dataset/monk2/monk2test_onehot.csv"

    training_set = get_dataset(train_csv_one_hot)
    test_set = get_dataset(test_csv_one_hot)

    eta = 0.3
    alpha = 0.2
    epochs = 100
    nn.train(training_set, epochs, eta, alpha)

    horror_plot(nn, eta, 0)


def run_model_cup_no_kfold():
    train_csv_one_hot = "dataset/monk1/monk1train_onehot.csv"
    test_csv_one_hot = "dataset/monk1/monk1test_onehot.csv"
    train_csv = "dataset/monk1/monk1train.csv"
    test_csv = "dataset/monk1/monk1test.csv"
    training_cup = "dataset/blindcup/LOC-OSM2-TR.csv"

    training_set = get_dataset(training_cup)
    test_set = get_dataset(training_cup)


def run_model_cup_kfold():
    training_cup = "dataset/blindcup/LOC-OSM2-TR.csv"


if __name__ == "__main__":
    test_monk()

