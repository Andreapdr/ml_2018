from neuralNetwork import NeuralNet
from utils import get_dataset, horror_plot

def main():
    # Standard Monk Dataset
    train_csv = "dataset/monk1/monk1train.csv"
    test_csv = "dataset/monk1/monk1test.csv"

    # One-Hot Encoded Monk Dataset (len = 16 + 1, target at index 0)
    train_csv_one_hot = "dataset/monk2/monk2train_onehot.csv"
    test_csv_one_hot = "dataset/monk2/monk2test_onehot.csv"

    training_set = get_dataset(train_csv_one_hot)
    test_set = get_dataset(test_csv_one_hot)
    # training_set = get_dataset(train_csv)
    # test_set = get_dataset(test_csv)

    ''' NB: every layer must have as many weights as the previous layer's neuron
        SET NETWORK STRUCTURE WITH APPROPRIATE WEIGHT AMOUNTS AND LAYERS.    
        Initialize empty network = list containing layers
        set a first in layer (c neuron, d weights each)
        set out_layer (e neuron, c weights each) '''

    # INITIALIZATION
    nn = NeuralNet()
    nn.initialize_layer(6, 17)
    nn.initialize_layer(6, 6)
    nn.initialize_layer(1, 6)

    # TRAINING SESSION
    lr = 0.25
    momentum = 0.0
    # TODO: Check alpha not working as intended
    alpha = 0.00
    nn.training(50, training_set, test_set, lr, momentum, alpha, verbose=False,
                step_decay=True, lr_decay=False)

    # SCREENING
    horror_plot(nn, lr, momentum)


if __name__ == "__main__":
    main()
