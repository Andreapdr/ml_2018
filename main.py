from neuralNetwork import NeuralNet
from utils import get_dataset, horror_plot, k_fold

""" lr = Learning Rate,
    alpha = Momentum
    step_decay = value multiplying learning rate every 20 epochs - should be lower than 1 """

# TODO: IMPLEMENT BIAS AND REGULARIZATION


def run_monk():
    nn = NeuralNet("mean_squared_error")
    nn.init_input_layer(17)
    nn.init_layer(3, 17, "sigmoid")
    nn.init_layer(1, 3, "sigmoid")

    train_csv_one_hot = "dataset/monk2/monk2train_onehot.csv"
    test_csv_one_hot = "dataset/monk2/monk2test_onehot.csv"

    training_set = get_dataset(train_csv_one_hot)
    validation_set = get_dataset(test_csv_one_hot)

    task = "monk"
    eta = 0.3
    alpha = 0.2
    epochs = 100
    nn.train(task, training_set, validation_set, epochs, eta, alpha)

    horror_plot(nn, eta, 0)


def run_cup():
    training_cup = "dataset/blindcup/LOC-OSM2-TR.csv"

    training_set = get_dataset(training_cup)
    validation_set = get_dataset(training_cup)

    nn = NeuralNet("mean_squared_error")
    nn.init_input_layer(10)
    nn.init_layer(23, 10, "tanh")
    nn.init_layer(23, 23, "tanh")
    nn.init_layer(23, 23, "tanh")
    nn.init_layer(2, 23, "linear")

    task = "cup"
    eta = 0.003
    alpha = 0.3
    epochs = 100
    nn.train(task, training_set, validation_set, epochs, eta, alpha)
    horror_plot(nn, eta, 0)


def run_model_cup_kfold():
    training_cup = "dataset/blindcup/LOC-OSM2-TR.csv"


if __name__ == "__main__":
    run_cup()

