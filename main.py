from neuralNetwork import NeuralNet
from utils import get_dataset, horror_plot, horror_plot2, k_fold

""" lr = Learning Rate,
    alpha = Momentum
    step_decay = value multiplying learning rate every 20 epochs - should be lower than 1 """


def run_model_no_kfold():
    train_csv_one_hot = "dataset/monk1/monk1train_onehot.csv"
    test_csv_one_hot = "dataset/monk1/monk1test_onehot.csv"
    train_csv = "dataset/monk1/monk1train.csv"
    test_csv = "dataset/monk1/monk1test.csv"
    testing_cup = "dataset/blindcup/LOC-OSM2-TR.csv"

    training_set = get_dataset(train_csv_one_hot)
    test_set = get_dataset(test_csv_one_hot)

    # NO KFOLD
    for i in range(1):
        nn = NeuralNet()
        nn.init_input_layer(17)
        nn.init_layer(4, 17, "sigmoid")
        nn.init_layer(1, 4, "sigmoid")

        lr = 0.3
        epochs = 150
        alpha = 0.2
        step_decay = 1
        nn.train(training_set, test_set, epochs, lr, alpha, step_decay)

    horror_plot(nn, lr, 0)


def run_model_kfold():
    train_csv_one_hot = "dataset/monk2/monk2train_onehot.csv"
    test_csv_one_hot = "dataset/monk2/monk2test_onehot.csv"
    train_csv = "dataset/monk1/monk1train.csv"
    test_csv = "dataset/monk1/monk1test.csv"
    testing_cup = "dataset/blindcup/LOC-OSM2-TR.csv"

    # NB: USING TEST SET TO TRAIN B/C MONK DATASET ...
    training_set = get_dataset(train_csv_one_hot)
    test_set = get_dataset(test_csv_one_hot)

    train_folded, val_folded = k_fold(test_set, 5)
    nn_to_plot = []
    for i in range(len(train_folded)):
        nn = NeuralNet()
        nn.init_input_layer(17)
        nn.init_layer(4, 17, "sigmoid")
        nn.init_layer(3, 4, "sigmoid")
        nn.init_layer(1, 3, "sigmoid")

        tr = train_folded[i]
        tval = val_folded[i]

        lr = 0.3
        epochs = 75
        alpha = 0.2
        step_decay = 1
        nn.train(tr, tval, epochs, lr, alpha, step_decay)
        nn_to_plot.append(nn)

    horror_plot2(nn_to_plot, lr, 0)


if __name__ == "__main__":
    run_model_no_kfold()

