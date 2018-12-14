from neuralNetwork import NeuralNet
from utils import get_dataset, horror_plot, horror_plot2, k_fold
import itertools
import time

""" lr = Learning Rate,
    alpha = Momentum
    step_decay = value multiplying learning rate every 20 epochs - should be lower than 1 """

# TODO: IMPLEMENT BIAS AND REGULARIZATION

# Setting model preferences
grid_search = True
task = "cup"
kfold = True
epochs = 50

# Setting hyperparameters manually - change if grid_search is False
eta = 0.3
alpha = 0.3

# Setting hyperparameters bounds for grid search - change if grid_search is True
eta_gs = [0.1, 0.2, 0.3]
alpha_gs = [0.1, 0.2, 0.3]


def run_test_monk():
    nn = NeuralNet("mean_squared_error")
    nn.init_input_layer(17)
    nn.init_layer(3, 17, "tanh")
    nn.init_layer(1, 3, "tanh")

    train_csv_one_hot = "dataset/monk2/monk2train_onehot.csv"
    test_csv_one_hot = "dataset/monk2/monk2test_onehot.csv"

    training_set = get_dataset(train_csv_one_hot)
    validation_set = get_dataset(test_csv_one_hot)

    nn.train(task, training_set, validation_set, epochs, eta, alpha, verbose)

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

    nn.train(task, training_set, validation_set, epochs, eta, alpha, verbose)
    horror_plot(nn, eta, 0)


def run_model_cup_kfold():
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
    hp = list()
    hp.append(eta_gs)
    hp.append(alpha_gs)

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
    verbose = True
    if grid_search:
        verbose = False
        run_grid_search()
    elif task == "monk":
        run_test_monk()
    else:
        if kfold:
            run_model_cup_kfold()
        else:
            run_cup()
