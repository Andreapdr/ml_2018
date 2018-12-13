import numpy as np
from layer import Layer
from inputLayer import InputLayer
import time


class NeuralNet:

    def __init__(self):
        self.layer_list = list()
        self.error_list = list()
        self.validation_error_list = list()
        self.accuracy_list = list()
        self.validation_accuracy_list = list()

    def init_input_layer(self, n_att):
        self.layer_list.append(InputLayer(n_att))

    def init_layer(self, n_neuron, n_weights, activation):
        self.layer_list.append(Layer(n_neuron, n_weights, activation))

    def feedforward(self, input_given):
        self.layer_list[0].init_input(input_given)      # inserting data in input layer
        for i in range(1, len(self.layer_list)):        # loop through layers starting from second one
            inputs = self.layer_list[i-1].neurons       # inputs are the neurons of previous layer
            self.layer_list[i].neurons = np.dot(inputs, self.layer_list[i].weights.T)   # compute neurons
            # self.layer_list[i].neurons = activation_function(self.layer_list[i].neurons + self.layer_list[i].bias)
            self.layer_list[i].neurons += self.layer_list[i].bias       # adding bias
            self.layer_list[i].activation_layer()       # calling activation function

    def compute_delta(self, target):
        for i in range(len(self.layer_list)-1, 0, -1):      # loop through layers from last one to second one
            # delta output_layer
            if i == len(self.layer_list)-1:
                # self.layer_list[i].delta = np.array(np.multiply(derivative_activation(self.layer_list[i].neurons),
                #                                                 (target - self.layer_list[i].neurons)))
                # compute delta by multiplying derivatives of neurons and loss
                self.layer_list[i].delta = np.array(np.multiply(self.layer_list[i].activation_function_derivative(),
                                                       (target - self.layer_list[i].neurons)))
            # delta hidden layers
            else:
                delta_upstream = self.layer_list[i+1].delta
                weights_upstream = self.layer_list[i+1].weights
                sum_delta_weights_upstream = np.dot(delta_upstream, weights_upstream)
                # temp = np.multiply(sum_delta_weights_upstream, derivative_activation(self.layer_list[i].neurons))
                # compute delta by multiplying delta of next layer and derivatives of neurons
                self.layer_list[i].delta = sum_delta_weights_upstream * self.layer_list[i].activation_function_derivative()

    def update_weights(self, learning_rate, epoch, alpha):
        for i in range(1, len(self.layer_list)):     # loop through layers from second one to last one
            # UPDATE BIAS FOR ENTIRE LAYER
            self.layer_list[i].bias += np.multiply(self.layer_list[i].delta, learning_rate)
            # UPDATE WEIGHTS CYCLING THROUGH LAYERS
            for j in range((len(self.layer_list[i].neurons))):   # loop as many times as the number of neurons
                weight_update = np.multiply(learning_rate, np.dot(self.layer_list[i].delta[j], self.layer_list[i - 1].neurons))
                if epoch != 0:
                    self.layer_list[i].previous_update[j] = weight_update
                    weight_update = weight_update + np.multiply(alpha, self.layer_list[i].previous_update[j])    # adding momentum
                    self.layer_list[i].weights[j] = self.layer_list[i].weights[j] + weight_update     # updating weights

    def train(self, training_set, validation_set, epoch, learning_rate, alpha, step_decay, verbose):
        for epoch in range(epoch):              # loop through epochs
            if verbose:
                time_start = time.clock()           # start counting time
            # np.random.shuffle(training_set)
            epoch_error = 0
            correct_pred = 0
            if epoch % 20 == 0 and epoch != 0:          # decrease learning rate every n steps
                learning_rate = learning_rate * step_decay
            if verbose:
                print(f"\nEPOCH {epoch+1} _______________________________________")
            for training_data in training_set:          # loop through training set
                training_input = training_data[1:]
                target = training_data[0]
                self.feedforward(training_input)
                self.compute_delta(target)
                self.update_weights(learning_rate, epoch, alpha)
                # error = 0.5 * mean_squared_error(target, self.layer_list[-1].neurons[0])
                error = 0.5 * (target - self.layer_list[-1].neurons[0]) ** 2
                # guess equal to 1 if the neuron in the output layers is greater then 0.5, 0 otherwise
                guess = 0
                if self.layer_list[-1].neurons > 0.5:
                    guess = 1
                # summing up correct predictions
                if (guess == target).all():
                    correct_pred += 1
                # summing up total error for epoch
                epoch_error += error
            if verbose:
                print(f"Total Error for Epoch on Training Set: {round(epoch_error/len(training_set), 5)}\n"
                      f"Accuracy on Training:   {round(correct_pred/len(training_set), 5)}")
            # adding error on the epoch to error list
            self.error_list.append((epoch+1, epoch_error/len(training_set)))
            # adding accuracy on the epoch to accuracy list
            self.accuracy_list.append((epoch+1, correct_pred/len(training_set)))
            self.test(validation_set, epoch+1, verbose)
            # compute time spent for epoch
            if verbose:
                time_elapsed = round((time.clock() - time_start), 3)
                print(f"Time elapsed for epoch {epoch+1}: {time_elapsed}s")

    def test(self, validation_set, relative_epoch, verbose):
        total_error = 0
        correct_pred = 0
        # np.random.shuffle(validation_set)
        for i in range(len(validation_set)):    # loop through validation set
            validation_in = validation_set[i][1:]
            target = validation_set[i][0]
            self.feedforward(validation_in)
            error = 0.5 * (target - self.layer_list[-1].neurons[0]) ** 2
            total_error += error
            # guess equal to 1 if the neuron in the output layers is greater then 0.5, 0 otherwise
            guess = 0
            if self.layer_list[-1].neurons > 0.5:
                guess = 1
            # summing up correct predictions
            if guess == target:
                correct_pred += 1
        # adding error on the epoch to error list
        self.validation_error_list.append((relative_epoch, total_error/len(validation_set)))
        # adding accuracy on the epoch to acccuracy list
        self.validation_accuracy_list.append((relative_epoch, correct_pred/len(validation_set)))
        if verbose:
            print(f"Total Error for Epoch on Validata Set: {round(total_error/len(validation_set), 5)}\n"
                  f"Accuracy on Validation: "
                  f"{round(correct_pred/len(validation_set), 5)}")

    # TODO REGULARIZATION to implement
    def regularization(self):
        pass

    # TODO to implement GridSearch and CrossValidation
    def cross_validation(self):
        pass

    # TODO GRID SEARCH
    def grid_search(self):
        pass


def derivative_sigmoid(x):
    return x * (1 - x)


def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))


def tanh_function(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - x**2


def mean_squared_error(target, output):
    return np.subtract(target, output) ** 2


def mean_euclidean_error(target, output):
    pass
