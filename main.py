from neuralNetwork import NeuralNet, init_manger
from utils import get_dataset, simple_plot, simple_plot_test, plot_multinetwork, k_fold
import itertools

# TODO: EARLY STOP plz

""" lr = Learning Rate,
    alpha = Momentum
    step_decay = value multiplying learning rate every 20 epochs - should be lower than 1 """


""" Simply run the model on monk dataset (no kfold)"""
def run_monk():
    nn = NeuralNet("mean_squared_error")
    nn.init_input_layer(17)
    nn.init_layer(3, 17, "sigmoid")
    nn.init_layer(1, 3, "sigmoid")

    train_csv_one_hot = "dataset/monk3/monk3train_onehot.csv"
    test_csv_one_hot = "dataset/monk3/monk3test_onehot.csv"

    training_set = get_dataset(train_csv_one_hot)
    validation_set = get_dataset(test_csv_one_hot)

    task = "monk"
    eta = 0.3
    alpha = 0.6
    lambd = 0.00
    eta_decay = 0.00
    epochs = 150
    nn.set_weights_pre_training()
    nn.train(task, training_set, validation_set, epochs, eta, alpha, lambd, eta_decay, True)

    simple_plot(task, nn, eta, alpha)


""" Simply run the model on cup dataset (no kfold)"""
def run_cup():
    training_cup = "dataset/blindcup/training_set.csv"

    training_set = get_dataset(training_cup)
    validation_set = get_dataset(training_cup)

    nn = NeuralNet("mean_euclidean_error")
    nn.init_input_layer(10)
    nn.init_layer(23, 10, "tanh")
    nn.init_layer(23, 23, "tanh")
    nn.init_layer(2, 23, "linear")

    task = "cup"
    eta = 0.01
    alpha = 0.80
    lambd = 0.00
    epochs = 250
    eta_decay = 0.00
    nn.train(task, training_set, validation_set, epochs, eta, alpha, lambd, eta_decay, True)
    simple_plot(task, nn, eta, alpha)


""" Run kfold validation on given dataset - then call for every folding run_cup_folded/run_monk_folded"""
def run_kfold_monk():
    training_set = "dataset/monk2/monk2train_onehot.csv"
    folds = 4
    train_folded, val_folded = k_fold(get_dataset(training_set), folds)
    nn_to_plot = []
    for i in range(len(train_folded)):
        model = run_monk_folded(train_folded[i], val_folded[i])
        nn_to_plot.append(model)

    # NB: ATM eta (in the plot title) is manually set to 0.3
    plot_multinetwork(nn_to_plot, 0.3, 0)


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
    eta = 0.003
    alpha = 0.2
    lambd = 0.01
    epochs = 100
    nn.train(task, training_set, validation_set, epochs, eta, alpha, False)
    return nn


def run_monk_folded():
    task = "monk"
    eta = 0.03
    alpha = 0.3
    lambd = 0.00
    eta_decay = 0.0
    epochs = 25
    verbose = True

    training_set = "dataset/monk3/monk3train_onehot.csv"
    test_set = "dataset/monk3/monk3test_onehot.csv"
    folds = 4
    train_folded, val_folded = k_fold(get_dataset(training_set), folds)
    nn_to_plot = []
    for i in range(len(train_folded)):
        nn = NeuralNet("mean_squared_error")
        nn.init_input_layer(17)
        nn.init_layer(3, 17, "sigmoid")
        nn.init_layer(1, 3, "sigmoid")

        tr = train_folded[i]
        tval = val_folded[i]

        nn.train(task, tr, tval, epochs, eta, alpha, lambd, eta_decay, verbose)
        nn_to_plot.append(nn)

        print(f"KFOLD {i + 1} of {folds} _______________________________________")

    plot_multinetwork(nn_to_plot, eta, alpha, lambd, folds, "monk")

    tr_f = get_dataset(training_set)
    ts_f = get_dataset(test_set)
    nn = NeuralNet("mean_squared_error")
    nn.init_input_layer(17)
    nn.init_layer(3, 17, "sigmoid")
    nn.init_layer(1, 3, "sigmoid")
    # # Training on whole training_set and "validating" on test_set
    nn.train(task, tr_f, ts_f, epochs, eta, alpha, lambd, eta_decay, verbose)
    simple_plot(task, nn, eta, alpha)


def run_model_cup_kfold():
    task = "cup"
    eta = 0.01
    alpha = 0.4
    lambd = 0.00
    eta_decay = 0.001
    epochs = 250
    verbose = False

    training_cup = "dataset/blindcup/training_set2.csv"
    folds = 1
    train_folded, val_folded = k_fold(get_dataset(training_cup), folds)
    nn_to_plot = []
    for i in range(len(train_folded)):
        # initializing network
        nn = NeuralNet("mean_euclidean_error")

        # initializing input layer
        nn.init_input_layer(10)

        # adding layers
        # nn.init_layer(20, 10, "tanh")
        # nn.init_layer(20, 20, "tanh")
        # nn.init_layer(20, 20, "tanh")
        # nn.init_layer(20, 20, "tanh")
        # nn.init_layer(2, 20, "linear")

        nn.init_layer(6, 10, "tanh")
        nn.init_layer(2, 6, "linear")

        # setting weights xavier init
        nn.set_weights_pre_training()

        tr = train_folded[i]
        tval = val_folded[i]

        nn.train(task, tr, tval, epochs, eta, alpha, lambd, eta_decay, verbose)
        nn_to_plot.append(nn)

        print(f"KFOLD {i+1} of {folds} _______________________________________")

    plot_multinetwork(nn_to_plot, eta, alpha, lambd, folds, "1hL*6N: tanh")


""" we should run a complete grid search for every network 
    architecture (hidden_layers/neuron per layer) we want to check...
    n_input: the number of attributes in a given dataset
    n_neurons_layer: the number of neurons for each hidden layer (assuming fully connected architecture)
    n_hidden_layers: the number of hidden layers contained in the network
    n_output_neurons: the number of output neurons """
def run_grid_search():
    epochs = 75
    n_input = [10]
    n_neurons_layer = [6, 8, 10, 20, 40]
    n_hidden_layers = [1, 2, 3, 4]
    n_output_neurons = [2]

    act_function_hidden = ["tanh"]
    act_function_output = ["linear"]
    error_function = ["mean_euclidean_error"]

    eta_swallow_gs = [0.3]
    alpha_swallow_gs = [0.3]
    lambd_swallow_gs = [0.01]

    eta_deep_gs = [0.0001, 0.001, 0.01, 0.1]
    # alpha_deep_gs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    alpha_deep_gs = [0.2, 0.4, 0.6, 0.8]
    lambd_deep_gs = [0.0001, 0.001, 0.01]

    hp_architecture = [n_hidden_layers, n_neurons_layer, n_input, n_output_neurons, error_function, act_function_hidden, act_function_output]
    hp_hyperparam = [eta_deep_gs, alpha_deep_gs, lambd_deep_gs]

    total_time_elapsed = 0
    best_error = 1000
    best_combination = [1, 1, 1]

    for j, architecture in enumerate(itertools.product(*hp_architecture)):
        for i, comb in enumerate(itertools.product(*hp_hyperparam)):
            # print(*architecture)
            task = "cup"
            eta = comb[0]
            alpha = comb[1]
            lambd = comb[2]
            eta_decay = 0.00
            verbose = False
            folds = 7
            nn_to_plot = []

            training_cup = "dataset/blindcup/training_set2.csv"
            train_folded, val_folded = k_fold(get_dataset(training_cup), folds)

            for i in range(len(train_folded)):
                nn = init_manger(*architecture)
                tr = train_folded[i]
                tval = val_folded[i]
                nn.train(task, tr, tval, epochs, eta, alpha, lambd, eta_decay, verbose)
                nn_to_plot.append(nn)
                print(f"KFOLD {i + 1} of {folds} _______________________________________")

            plot_multinetwork(nn_to_plot, eta, alpha, lambd, folds, architecture)

            error_kfold_tr = 0
            error_kfold_val = 0
            for network in nn_to_plot:
                error_kfold_tr += network.error_list[-1][1]
                error_kfold_val += network.validation_error_list[-1][1]

            error_kfold_tr = round(error_kfold_tr/len(train_folded), 5)
            error_kfold_val = round(error_kfold_val/len(train_folded), 5)
            print(f"NN Architecture: Layers: {len(nn_to_plot[0].layer_list)}, Units x layer: {len(nn_to_plot[0].layer_list[1].net)}"
                  f"\nHyperparameters: eta: {eta}, alpha: {alpha}, lambda: {lambd}, eta decay: {eta_decay}"
                  f"\nAverage final error on training set: {error_kfold_tr}"
                  f"\nAverage final error on validat. set: {error_kfold_val}")




    # for combination in itertools.product(*hp):
    #     lr = combination[0]
    #     alpha = combination[1]
    #
    #     time_start = time.perf_counter()
    #
    #     error = run_model_cup_kfold()
    #     if error < best_error:
    #         best_error = error
    #         best_combination = combination
    #     time_elapsed = time.perf_counter() - time_start
    #     total_time_elapsed += time_elapsed
    #     time_elapsed = round(time_elapsed, 3)
    #     print(
    #         f"Time elapsed for combination lr = {lr}, alpha = {alpha}: {time_elapsed}s\n"
    #         f"Final error: {error}\n")
    # print(f"Total time elapsed for grid search with {epochs} epochs: {round(total_time_elapsed, 3)}s\n"
    #       f"Best combination of hyperparameters: lr = {best_combination[0]}, "
    #       f"alpha = {best_combination[1]}\n"
    #       f"with error = {best_error}")


def test_init_manager():
    nn = init_manger(2, 5, 10, 2, "mean_euclidean_error", "tanh", "linear")
    task = "cup"
    eta = 0.001
    alpha = 0.9
    lambd = 0.01
    epochs = 100
    verbose = True
    training_cup = "dataset/blindcup/training_set.csv"
    folds = 1
    nn_to_plot = []

    train_folded, val_folded = k_fold(get_dataset(training_cup), folds)

    for i in range(len(train_folded)):
        tr = train_folded[i]
        tval = val_folded[i]
        nn.train(task, tr, tval, epochs, eta, alpha, lambd, verbose)
        nn_to_plot.append(nn)
        print(f"KFOLD {i + 1} of {folds} _______________________________________")

    plot_multinetwork(nn_to_plot, eta, 0, folds, 0)


if __name__ == "__main__":
    run_grid_search()
    # run_model_cup_kfold()
    # run_monk_folded()
    # run_monk()
