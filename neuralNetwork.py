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
        self.validation_error_list = list()
        self.validation_accuracy_list = list()

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

    def feedforward(self, input_prev, task):
        layers = self.layer_list
        layers[0].init_input(input_prev)
        for i in range(1, len(layers)):
            input_prev = layers[i - 1].out
            layers[i].net = np.dot(input_prev, layers[i].weights.T)
            layers[i].net += layers[i].bias_W
            layers[i].out = layers[i].activation_function()
            if task == "cup":
                layers[i].out = np.around(layers[i].out, 6)

    def compute_delta(self, target):
        layers = self.layer_list
        """DELTA OUTPUT LAYER"""
        deriv_err_out = self.loss_function(target, derivative=True)
        deriv_act = layers[-1].activation_function(derivative=True)
        layers[-1].delta = deriv_act * deriv_err_out
        error_output = self.loss_function(target)

        for i in range(len(self.layer_list) - 2, 0, -1):
            """DELTA HIDDEN LAYERS"""
            deriv_act = layers[i].activation_function(derivative=True)
            delta_upstream = layers[i+1].delta
            weights_upstream = layers[i+1].weights
            sum_delta_weights = np.dot(weights_upstream.T, delta_upstream)
            self.layer_list[i].delta = sum_delta_weights * deriv_act

        return error_output

    def update_weights(self, eta, alpha, lambd):
        layers = self.layer_list

        for i in range(1, len(layers)):
            layers[i].bias_W += self.layer_list[i].delta * eta

            momentum = layers[i].last_deltaW * alpha

            # TODO: REGULARIZATION TERM
            reg_term = (lambd * layers[i].weights)

            previous_input = np.array([layers[i-1].out])
            deltas_layer = np.array([layers[i].delta])
            delta_weights = np.dot(previous_input.T, deltas_layer) * eta
            delta_weights += momentum

            layers[i].weights += delta_weights.T - (2 * reg_term)
            layers[i].last_deltaW = delta_weights

    def train(self, task, training_set, validation_set, epochs, eta, alpha, lambd, verbose):
        for epoch in range(epochs):
            if verbose:
                time_start = time.clock()
            np.random.shuffle(training_set)
            epoch_error = 0
            correct_pred = 0
            if verbose:
                print(f"\nEPOCH {epoch + 1} _______________________________________")
            for training_data in training_set:
                if task == "monk":
                    target_train = training_data[0]
                    training_input = training_data[1:]
                elif task == "cup":
                    bound = len(training_data)
                    training_input = training_data[1:bound-2]
                    target_train = training_data[bound-2:]
                self.feedforward(training_input, task)
                error_out = self.compute_delta(target_train)
                epoch_error += np.sum(error_out)
                self.update_weights(eta, alpha, lambd)
                guess = self.layer_list[-1].out

                if task == "monk":
                    guess = 0
                    pred_temp = self.layer_list[-1].out
                    if pred_temp > 0.5:
                        guess = 1

                res = np.sum(np.subtract(target_train, guess))
                if res == 0:
                    correct_pred += 1
            if verbose:
                print(f"Total Error for Epoch on Training Set: {round(epoch_error / len(training_set), 5)}\n")
                if task == "monk":
                    print(f"Accuracy on Training:   {round(correct_pred / len(training_set), 5)}")
                time_elapsed = round((time.clock() - time_start), 3)
                print(f"Time elapsed for epoch {epoch + 1}: {time_elapsed}s")
            self.error_list.append((epoch + 1, epoch_error / len(training_set)))
            self.test(task, validation_set, epoch+1, verbose)

    def test(self, task, validation_set, relative_epoch, verbose):
        total_error = 0
        correct_pred = 0
        for i in range(len(validation_set)):
            if task == "monk":
                validation_in = validation_set[i][1:]
                target = validation_set[i][0]
            else:
                bound = len(validation_set[i])
                validation_in = validation_set[i][1:bound-2]
                target = validation_set[i][bound-2:]
            self.feedforward(validation_in, task)
            error = self.loss_function(target)
            total_error += np.sum(error)

            if task == "monk":
                guess = 0
                pred_temp = self.layer_list[-1].out
                if pred_temp > 0.5:
                    guess = 1
                res = np.sum(np.subtract(target, guess))
                if res == 0:
                    correct_pred += 1
                self.validation_accuracy_list.append((relative_epoch, correct_pred / len(validation_set)))
        self.validation_error_list.append((relative_epoch, total_error/len(validation_set)))
        if verbose:
            print(f"Total Error for Epoch on Validate Set: {round(total_error/len(validation_set), 5)}\n")
            if task == "monk":
                print(f"Accuracy on Validation: {round(correct_pred/len(validation_set), 5)}")


def mean_squared_error(target, output, derivative):
    if derivative:
        return np.subtract(target, output)
    else:
        res = np.subtract(target, output) ** 2
        res = np.sum(res, axis=0)
        return res/len(output)


# TODO: WHY does it (only) converge if we return - derivative ?
def mean_euclidean_error(target_value, neurons_out, derivative):
    if derivative:
        err = mean_euclidean_error(target_value, neurons_out, derivative=False)
        return - np.subtract(neurons_out, target_value) * (1 / err)
    res = np.subtract(neurons_out, target_value) ** 2
    res = np.sum(res, axis=0)
    res = np.sqrt(res)
    return res / len(neurons_out)
