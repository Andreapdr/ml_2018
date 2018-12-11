import numpy as np
import time
from layer import Layer
from inputLayer import InputLayer


class NeuralNet:

    def __init__(self, loss):
        self.loss = None
        self.set_loss_function(loss)
        self.layer_list = list()
        self.error_list = list()

    def set_loss_function(self, loss):
        if loss == "mean_squared_error":
            self.loss = mean_squared_error
        elif loss == "mean_euclidean_error":
            self.loss = mean_euclidean_error

    def loss_function(self, target, derivative=False):
        return self.loss(target, self.layer_list[-1].out, derivative)

    def init_input_layer(self, n_att):
            self.layer_list.append(InputLayer(n_att))

    def init_layer(self, n_neuron, n_weights, activation):
            self.layer_list.append(Layer(n_neuron, n_weights, activation))

    def feedforward(self, input):
        layers = self.layer_list
        layers[0].init_input(input)
        for i in range(1, len(layers)):
            input = layers[i-1].out
            layers[i].net = np.dot(input, layers[i].weights.T)
            layers[i].out = layers[i].activation_function()

    def compute_delta(self, target):
        layers = self.layer_list
        """DELTA OUTPUT LAYER"""
        deriv_err_out = self.loss_function(target, derivative=True)
        deriv_act = layers[-1].activation_function(derivative=True)
        layers[-1].delta = deriv_act * deriv_err_out
        # error_output = 0.5 * (target - layers[-1].out)**2
        # TODO: SHOULD I DIVIDE BY 0.5 OR NOT? Doesnt really change anything though ...
        error_output = self.loss_function(target)

        for i in range(len(self.layer_list) - 2, 0, -1):
            """DELTA HIDDEN LAYERS"""
            deriv_act = layers[i].activation_function(derivative=True)
            delta_upstream = layers[i+1].delta
            weights_upstream = layers[i+1].weights
            sum_delta_weights = np.dot(weights_upstream.T, delta_upstream)
            self.layer_list[i].delta = sum_delta_weights * deriv_act

        return error_output

    def update_weights(self, eta, alpha):
        layers = self.layer_list

        for i in range(1, len(layers)):
            momentum = layers[i].last_deltaW * alpha

            previous_input = np.array([layers[i-1].out])
            deltas_layer = np.array([layers[i].delta])
            delta_weights = np.dot(previous_input.T, deltas_layer) * eta
            delta_weights += momentum

            layers[i].weights += delta_weights.T
            layers[i].last_deltaW = delta_weights

    def train(self, training_set, epochs, eta, alpha):
        for epoch in range(epochs):
            time_start = time.clock()
            np.random.shuffle(training_set)
            epoch_error = 0
            correct_pred = 0
            print(f"\nEPOCH {epoch + 1} _______________________________________")
            for training_data in training_set:
                target_train = training_data[0]
                training_input = training_data[1:]
                self.feedforward(training_input)
                error_out = self.compute_delta(target_train)
                epoch_error += np.sum(error_out)
                self.update_weights(eta, alpha)

                guess = 0
                pred_temp = self.layer_list[-1].out
                if pred_temp > 0.5:
                    guess = 1

                res = np.sum(np.subtract(target_train, guess))
                if res == 0:
                    correct_pred += 1

            print(f"Total Error for Epoch on Training Set: {round(epoch_error / len(training_set), 5)}\n"
                  f"Accuracy on Training:   {round(correct_pred / len(training_set), 5)}")
            time_elapsed = round((time.clock() - time_start), 3)
            print(f"Time elapsed for epoch {epoch + 1}: {time_elapsed}s")
            self.error_list.append((epoch + 1, epoch_error / len(training_set)))


def mean_squared_error(target, output, derivative):
    if derivative:
        return np.subtract(target, output)
    else:
        res = np.subtract(target, output) ** 2
        res = np.sum(res, axis=0)
        res = np.sum(res, axis=0)
        return res/len(output)


# TODO CHECK HERE
def mean_euclidean_error(target_value, neurons_out, deriv=False):
    if deriv:
        err = mean_euclidean_error(target_value, neurons_out)
        return np.subtract(neurons_out, target_value) * (1 / err)
    res = np.subtract(neurons_out, target_value) ** 2
    res = np.sum(res, axis=0)
    res = np.sqrt(res)
    res = np.sum(res, axis=0)
    return (res / target_value.shape[1])
