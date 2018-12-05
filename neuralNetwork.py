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

    def init_layer(self, n_neuron, n_weights):
        self.layer_list.append(Layer(n_neuron, n_weights))

    def feedforward(self, input_given, activation_function):
        self.layer_list[0].init_input(input_given)
        for i in range(1, len(self.layer_list)):
            inputs = self.layer_list[i-1].neurons
            self.layer_list[i].neurons = np.dot(inputs, self.layer_list[i].weights.T)
            self.layer_list[i].neurons = activation_function(self.layer_list[i].neurons + self.layer_list[i].bias)

    def compute_delta(self, target, derivative_activation):
        for i in range(len(self.layer_list)-1, 0, -1):
            # delta output_layer
            if i == len(self.layer_list)-1:
                self.layer_list[i].delta = np.array(np.multiply(derivative_activation(self.layer_list[i].neurons),
                                                                (target - self.layer_list[i].neurons)))
            # delta hidden layers
            else:
                delta_upstream = self.layer_list[i+1].delta
                weights_upstream = self.layer_list[i+1].weights
                sum_delta_weights_upstream = np.dot(delta_upstream, weights_upstream)
                temp = np.multiply(sum_delta_weights_upstream, derivative_activation(self.layer_list[i].neurons))
                self.layer_list[i].delta = temp

    def update_weights(self, learning_rate):
        for i in range(1, len(self.layer_list)):
            # UPDATE BIAS FOR ENTIRE LAYER
            self.layer_list[i].bias += np.multiply(self.layer_list[i].delta, learning_rate)
            for j in range((self.layer_list[i].weights.shape[0])):
                temp = self.layer_list[i].weights[j] + np.multiply(learning_rate, np.dot(self.layer_list[i].delta[j],
                                                                                         self.layer_list[i - 1].neurons))
                self.layer_list[i].weights[j] = temp

    def train(self, training_set, validation_set, epoch, learning_rate, activation_function, derivative_activation):
        for epoch in range(epoch):
            time_start = time.clock()
            # np.random.shuffle(training_set)
            epoch_error = 0
            correct_pred = 0
            print(f"\nEPOCH {epoch+1} _______________________________________")
            for training_data in training_set:
                target = training_data[0]
                training_input = training_data[1:]
                self.feedforward(training_input, activation_function)
                self.compute_delta(target, derivative_activation)
                self.update_weights(learning_rate)
                guess = 0
                error = 0.5 * ((target - np.sum(self.layer_list[-1].neurons))**2)
                if self.layer_list[-1].neurons > 0.5:
                    guess = 1
                if (guess == target).all():
                    correct_pred += 1
                epoch_error += np.sum(error)
            print(f"Total Error for Epoch on Training Set: {round(epoch_error/len(training_set), 5)}\n"
                  f"Accuracy on Training:   {round(correct_pred/len(training_set), 5)}")
            self.error_list.append((epoch+1, epoch_error/len(training_set)))
            self.accuracy_list.append((epoch+1, correct_pred/len(training_set)))
            self.test(validation_set, epoch+1, activation_function)
            time_elapsed = round((time.clock() - time_start), 3)
            print(f"Time elapsed for epoch {epoch+1}: {time_elapsed}s")
            # if correct_pred/len(training_set) < 0.5:
            #     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    def test(self, validation_set, relative_epoch, activation_function):
        total_error = 0
        correct_pred = 0
        # np.random.shuffle(validation_set)
        for i in range(len(validation_set)):
            validation_in = validation_set[i][1:]
            target = validation_set[i][0]
            self.feedforward(validation_in, activation_function)
            error = 0.5 * ((target - np.sum(self.layer_list[-1].neurons))**2)
            total_error += error
            guess = 0
            if self.layer_list[-1].neurons > 0.5:
                guess = 1
            if guess == target:
                correct_pred += 1
        self.validation_error_list.append((relative_epoch, total_error/len(validation_set)))
        self.validation_accuracy_list.append((relative_epoch, correct_pred/len(validation_set)))
        print(f"Total Error for Epoch on Validata Set: {round(total_error/len(validation_set), 5)}\n"
              f"Accuracy on Validation: "
              f"{round(correct_pred/len(validation_set), 5)}")

    def get_number_neurons(self):
        tot = 0
        for i in range(1, len(self.layer_list)):
            tot += len(self.layer_list[i].neurons)
        return tot

    def get_number_weights(self):
        tot = 0
        for i in range(1, len(self.layer_list)):
            tot += self.layer_list[i].weights.shape[1] * len(self.layer_list[i].neurons)
        return tot

    # TODO MOMENTUM to implement
    def momentum(self):
        pass

    # TODO REGULARIZATION to implement
    def regularization(self):
        pass

    # TODO more ERROR FUNCTION to implement other error functions
    def error_function(self):
        pass

    # TODO to implement GridSearch and CrossValidation
    def cross_validation(self):
        pass

    def grid_search(self):
        pass

#     TODO implement WEIGHT DECAY!


def derivative_sigmoid(x):
    return x * (1 - x)


def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))


def tanh_function(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - x**2
