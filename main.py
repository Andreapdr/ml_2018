from neuralNetwork import NeuralNet, sigmoid_function, derivative_sigmoid, tanh_function, tanh_derivative
from utils import get_dataset, horror_plot, k_fold

""" TODO: implement the possibility to have different activation functions wrt to the layer (for ex:
    all hidden layers with tanh and output linear act). NB: when computing delta/gradient the different activation
    function SHOULD BE TAKEN INTO ACCOUNT - implement a layer's attribute specifying the activation function """

""" lr = Learning Rate,
    alpha = Momentum
    step_decay = value multiplying learning rate every 20 epochs - should be lower than 1 """


def main():
    train_csv_one_hot = "dataset/monk2/monk2train_onehot.csv"
    test_csv_one_hot = "dataset/monk2/monk2test_onehot.csv"
    train_csv = "dataset/monk1/monk1train.csv"
    test_csv = "dataset/monk1/monk1test.csv"
    testing_cup = "dataset/blindcup/LOC-OSM2-TR.csv"

    training_set = get_dataset(train_csv_one_hot)
    test_set = get_dataset(test_csv_one_hot)

    train_folded, val_folded = k_fold(test_set, 2)
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
        epochs = 100
        alpha = 0.2
        step_decay = 1
        activation_function = sigmoid_function
        derivative_activation = derivative_sigmoid
        nn.train(tr, tval, epochs, lr, alpha, step_decay, activation_function, derivative_activation)
        nn_to_plot.append(nn)

    horror_plot(nn_to_plot, lr, 0)


if __name__ == "__main__":
    main()

