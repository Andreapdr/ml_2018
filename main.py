from neuralNetwork import NeuralNet
from utils import get_dataset, horror_plot, horror_plot2, k_fold
import itertools
import time

""" lr = Learning Rate,
    alpha = Momentum
    step_decay = value by which the learning rate is multiplied every 20 epochs - should be lower than 1 """

# Importing dataset
train_csv_one_hot = "dataset/monk2/monk2train_onehot.csv"
test_csv_one_hot = "dataset/monk2/monk2test_onehot.csv"
train_csv = "dataset/monk1/monk1train.csv"
test_csv = "dataset/monk1/monk1test.csv"
testing_cup = "dataset/blindcup/LOC-OSM2-TR.csv"

# NB: USING TEST SET TO TRAIN B/C MONK DATASET ...
training_set = get_dataset(train_csv_one_hot)
test_set = get_dataset(test_csv_one_hot)


grid_search = True

# Setting hyperparameters manually - change if grid_search is False
lr = 0.3
epochs = 50
alpha = 0.2
step_decay = 1
kfold = False

# Setting hyperparameters bounds for grid search - change if grid_search is True

lr_gs = [0.1, 0.2, 0.3]
alpha_gs = [0.1, 0.2, 0.3]
step_decay_gs = [0.8, 0.9, 1]


def run_grid_search():
    hp = list()
    hp.append(lr_gs)
    hp.append(alpha_gs)
    hp.append(step_decay_gs)
    total_time_elapsed = 0
    best_error = 100
    best_combination = [1, 1, 1]

    for combination in itertools.product(*hp):
        global lr, alpha, step_decay
        lr = combination[0]
        alpha = combination[1]
        step_decay = combination[2]

        time_start = time.perf_counter()

        error = run_model_no_kfold()
        if error < best_error:
            best_error = error
            best_combination = combination
        time_elapsed = time.perf_counter() - time_start
        total_time_elapsed += time_elapsed
        time_elapsed = round(time_elapsed, 3)
        print(f"Time elapsed for combination lr = {lr}, alpha = {alpha}, step decay = {step_decay}: {time_elapsed}s\n"
              f"Final error: {error}\n")
    print(f"Total time elapsed for grid search: {round(total_time_elapsed, 3)}s\n"
          f"Best combination of hyperparameters: lr = {best_combination[0]}, "
          f"alpha = {best_combination[1]}, step_decay = {best_combination[2]}\n"
          f"with error = {best_error}")


# Model without K-fold
def run_model_no_kfold():
    nn = NeuralNet()                # initializing network
    nn.init_input_layer(17)         # initializing input layer

    # adding layers
    nn.init_layer(4, 17, "sigmoid")
    nn.init_layer(1, 4, "sigmoid")

    nn.train(training_set, test_set, epochs, lr, alpha, step_decay, verbose)     # training network
    if grid_search:
        return nn.validation_error_list[-1][1]
    horror_plot(nn, lr, 0)          # screening


# Model with K-fold
def run_model_kfold():
    x = test_set
    train_folded, val_folded = k_fold(test_set, 5)      # split dataset in k folds. For each fold 1/5 is used as val
    nn_to_plot = []
    for i in range(len(train_folded)):
        nn = NeuralNet()            # initializing network
        nn.init_input_layer(17)     # initializing input layer

        # adding layers
        nn.init_layer(4, 17, "sigmoid")
        nn.init_layer(3, 4, "sigmoid")
        nn.init_layer(1, 3, "sigmoid")

        tr = train_folded[i]
        tval = val_folded[i]

        nn.train(tr, tval, epochs, lr, alpha, step_decay, verbose)
        nn_to_plot.append(nn)   # add neural network to an array to be plotted

    horror_plot2(nn_to_plot, lr, 0)     # screening


if __name__ == "__main__":
    if grid_search:
        verbose = False
        run_grid_search()
    else:
        verbose = True
        if kfold:
            run_model_kfold()
        else:
            run_model_no_kfold()

