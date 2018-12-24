from neuralNetwork import NeuralNet
from utils import get_dataset, horror_plot, horror_plot2, k_fold
import itertools
import time

""" lr = Learning Rate,
    alpha = Momentum
    step_decay = value multiplying learning rate every 20 epochs - should be lower than 1 """

# TODO: check REGULARIZATION and implement STEP (LR) DECAY

""" Simply run the model on monk dataset (no kfold)"""
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
    lambd = 0.00
    epochs = 75
    nn.train(task, training_set, validation_set, epochs, eta, alpha, lambd, True)
    horror_plot(nn, eta, 0)


""" Simply run the model on cup dataset (no kfold"""
def run_cup():
    training_cup = "dataset/blindcup/LOC-OSM2-TR.csv"

    training_set = get_dataset(training_cup)
    validation_set = get_dataset(training_cup)

    nn = NeuralNet("mean_euclidean_error")
    nn.init_input_layer(10)
    nn.init_layer(23, 10, "tanh")
    nn.init_layer(23, 23, "tanh")
    nn.init_layer(23, 23, "tanh")
    nn.init_layer(2, 23, "linear")

    task = "cup"
    eta = 0.003
    alpha = 0.3
    lambd = 0.01
    epochs = 150
    nn.train(task, training_set, validation_set, epochs, eta, alpha, lambd, True)
    horror_plot(nn, eta, 0)


""" Run kfold validation on given dataset - then call for every folding run_cup_folded/run_monk_folded"""
def run_kfold():
    training_set = "dataset/monk2/monk2train_onehot.csv"
    folds = 5
    train_folded, val_folded = k_fold(get_dataset(training_set), folds)
    nn_to_plot = []
    for i in range(len(train_folded)):
        model = run_monk_folded(train_folded[i], val_folded[i])
        nn_to_plot.append(model)

    """LR nel plot settato manualmente al momento... (0.3)"""
    horror_plot2(nn_to_plot, 0.3, 0)


def run_monk_folded(train_set, val_set):
    nn = NeuralNet("mean_squared_error")
    nn.init_input_layer(17)
    nn.init_layer(3, 17, "sigmoid")
    nn.init_layer(1, 3, "sigmoid")

    training_set = train_set
    validation_set = val_set

    task = "monk"
    eta = 0.3
    alpha = 0.2
    lambd = 0.00
    epochs = 100
    nn.train(task, training_set, validation_set, epochs, eta, alpha, False)
    return nn


def run_cup_folded(train_set, val_set):
    nn = NeuralNet("mean_squared_error")
    nn.init_input_layer(10)
    nn.init_layer(23, 10, "tanh")
    nn.init_layer(23, 23, "tanh")
    nn.init_layer(23, 23, "tanh")
    nn.init_layer(2, 23, "linear")

    training_set = train_set
    validation_set = val_set

    task = "cup"
    eta = 0.3
    alpha = 0.2
    lambd = 0.01
    epochs = 100
    nn.train(task, training_set, validation_set, epochs, eta, alpha, False)
    return nn


def run_model_cup_kfold():
    task = None
    epochs = None
    eta = None
    alpha = None
    lambd = None
    verbose = None
    grid_search = None

    training_cup = "dataset/blindcup/LOC-OSM2-TR.csv"
    folds = 5
    train_folded, val_folded = k_fold(get_dataset(training_cup), folds)
    nn_to_plot = []
    sum_val_error = 0.0
    for i in (range(len(train_folded))):
        nn = NeuralNet("mean_squared_error")  # initializing network
        nn.init_input_layer(10)  # initializing input layer

        # adding layers
        nn.init_layer(3, 10, "sigmoid")
        nn.init_layer(3, 3, "sigmoid")
        nn.init_layer(2, 3, "sigmoid")

        tr = train_folded[i]
        tval = val_folded[i]

        nn.train(task, tr, tval, epochs, eta, alpha, verbose)
        nn_to_plot.append(nn)  # add neural network to an array to be plotted
        sum_val_error += nn.validation_error_list[-1][1]

    if grid_search:
        mean_val_error = sum_val_error / folds
        return mean_val_error
    horror_plot2(nn_to_plot, eta, 0)  # screening


def run_grid_search():
    epochs = None
    eta_gs = [0.1, 0.2, 0.3]
    alpha_gs = [0.1, 0.2, 0.3]
    lambd_gs = [0.01, 0.02, 0.03]

    hp = list()
    hp.append(eta_gs)
    hp.append(alpha_gs)
    hp.append(lambd_gs)

    total_time_elapsed = 0
    best_error = 1000
    best_combination = [1, 1, 1]

    for combination in itertools.product(*hp):
        lr = combination[0]
        alpha = combination[1]

        time_start = time.perf_counter()

        error = run_model_cup_kfold()
        if error < best_error:
            best_error = error
            best_combination = combination
        time_elapsed = time.perf_counter() - time_start
        total_time_elapsed += time_elapsed
        time_elapsed = round(time_elapsed, 3)
        print(
            f"Time elapsed for combination lr = {lr}, alpha = {alpha}: {time_elapsed}s\n"
            f"Final error: {error}\n")
    print(f"Total time elapsed for grid search with {epochs} epochs: {round(total_time_elapsed, 3)}s\n"
          f"Best combination of hyperparameters: lr = {best_combination[0]}, "
          f"alpha = {best_combination[1]}\n"
          f"with error = {best_error}")


if __name__ == "__main__":
    run_cup()
