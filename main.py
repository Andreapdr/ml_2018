import numpy as np
from neuralNetwork import NeuralNet, sigmoid_function, derivative_sigmoid, tanh_function, tanh_derivative
import matplotlib.pyplot as plt


def get_dataset(name_csv):
    dataset_np = np.genfromtxt(name_csv, delimiter=",")
    return dataset_np


train_csv_one_hot = "dataset/monk2/monk2train_onehot.csv"
test_csv_one_hot = "dataset/monk2/monk2test_onehot.csv"
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

neurons = nn.get_number_neurons()
print(neurons)
weights = nn.get_number_weights()
print(weights)
nn.train(training_set, test_set, 150, 0.25, tanh_function, tanh_derivative)


cord_x = list()
cord_y = list()
cord_x_val = list()
cord_y_val = list()
for elem in nn.error_list:
    cord_x.append(elem[0])
    cord_y.append(elem[1])
for elem in nn.validation_error_list:
    cord_x_val.append(elem[0])
    cord_y_val.append(elem[1])
plt.plot(cord_x, cord_y, label="Error Rate Training")
plt.plot(cord_x_val, cord_y_val, label="Error Rate Validation")
plt.legend()
plt.grid(True)
plt.show()
