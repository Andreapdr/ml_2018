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
        self.accuracy_list = list()
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

    def set_weights_pre_training(self):
        layer_list = self.layer_list
        for i in range(1, len(layer_list)):
            n_in = int(layer_list[i - 1].out.shape[0])
            desired_var = 2 / n_in
            layer_list[i].set_weights_xavier(desired_var)

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
            delta_upstream = layers[i + 1].delta
            weights_upstream = layers[i + 1].weights
            sum_delta_weights = np.dot(weights_upstream.T, delta_upstream)
            self.layer_list[i].delta = sum_delta_weights * deriv_act

        return error_output

    def update_weights(self, eta, alpha, lambd):
        layers = self.layer_list

        for i in range(1, len(layers)):
            layers[i].bias_W += self.layer_list[i].delta * eta

            momentum = layers[i].last_deltaW * alpha
            reg_term = lambd * layers[i].weights.T

            previous_input = np.array([layers[i - 1].out])
            deltas_layer = np.array([layers[i].delta])
            delta_weights = np.dot(previous_input.T, deltas_layer) * eta + momentum - reg_term

            layers[i].weights += delta_weights.T
            layers[i].last_deltaW = delta_weights

    def train(self, task, training_set, validation_set, epochs, eta, alpha, lambd, eta_decay, verbose, grid_search):
        for epoch in range(epochs):
            time_start = time.clock()
            total_time = time.clock()
            np.random.shuffle(training_set)
            epoch_error = 0
            correct_pred = 0
            if eta_decay and epoch != 0 and epoch % 25 == 0:
                eta = eta - eta * eta_decay
            if verbose:
                print(f"\nEPOCH {epoch + 1} _______________________________________")
            for training_data in training_set:
                if task == "monk":
                    target_train = training_data[0]
                    training_input = training_data[1:]
                elif task == "cup":
                    bound = len(training_data)
                    training_input = training_data[1:bound - 2]
                    target_train = training_data[bound - 2:]
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

            # Printing Results
            if verbose and epoch != epochs:
                print(f"Total Error for Epoch on Training Set: {round(epoch_error / len(training_set), 5)}")
                if task == "monk":
                    print(f"\nAccuracy on Training:   {round(correct_pred / len(training_set), 5)}")
                time_elapsed = round((time.clock() - time_start), 3)
                print(f"Time elapsed for epoch {epoch + 1}: {time_elapsed}s")

            # Last Epoch --> print Error
            if epoch == epochs - 1 and grid_search == False:
                print(f"Final Results:\n"
                      f"NN Architecture: Layers: {len(self.layer_list)}, "
                      f"Units x Hlayer: {len(self.layer_list[1].net)}")
                print(f"Total Error for Epoch on Training Set: {round(epoch_error / len(training_set), 5)}")
                if task == "monk":
                    print(f"\nAccuracy on Training:   {round(correct_pred / len(training_set), 5)}")
                total_time_elapsed = round((time.clock() - total_time), 3)

            self.error_list.append((epoch + 1, round(epoch_error / len(training_set), 5)))
            self.accuracy_list.append((epoch + 1, round(correct_pred / len(training_set))))
            self.test(task, validation_set, epoch + 1, verbose, epochs, grid_search)
            # Last Epoch --> print Total Time
            if epoch == epochs - 1 and verbose:
                print(f"Total time elapsed: {total_time_elapsed}s")

    def test(self, task, validation_set, relative_epoch, verbose, epochs, grid_search):
        total_error = 0
        correct_pred = 0
        np.random.shuffle(validation_set)
        for i in range(len(validation_set)):
            if task == "monk":
                validation_in = validation_set[i][1:]
                target = validation_set[i][0]
            else:
                bound = len(validation_set[i])
                validation_in = validation_set[i][1:bound - 2]
                target = validation_set[i][bound - 2:]
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

        # Printing Results
        self.validation_error_list.append((relative_epoch, round(total_error / len(validation_set), 5)))
        self.validation_accuracy_list.append((relative_epoch, round(correct_pred / len(validation_set), 5)))
        if verbose:
            print(f"Total Error for Epoch on Validate Set: {round(total_error / len(validation_set), 5)}")
            if task == "monk":
                print(f"\nAccuracy on Validation: {round(correct_pred / len(validation_set), 5)}")
        elif relative_epoch == epochs and grid_search == False:
            print(f"Total Error for Epoch on Validate Set: {round(total_error / len(validation_set), 5)}")
            if task == "monk":
                print(f"\nAccuracy on Validation: {round(correct_pred / len(validation_set), 5)}")

    def train_final(self, training_set, epochs, eta, alpha, lambd, eta_decay, verbose):
        total_time = time.clock()
        first = True
        for epoch in range(epochs):
            time_start = time.clock()
            np.random.shuffle(training_set)
            epoch_error = 0
            if eta_decay and epoch != 0 and epoch % 25 == 0:
                eta = eta - eta * eta_decay
            if verbose:
                print(f"\nEPOCH {epoch + 1} _______________________________________")
            for training_data in training_set:
                bound = len(training_data)
                training_input = training_data[1:bound - 2]
                target_train = training_data[bound - 2:]
                self.feedforward(training_input, "cup")
                error_out = self.compute_delta(target_train)
                epoch_error += np.sum(error_out)
                self.update_weights(eta, alpha, lambd)

            time_elapsed = round((time.clock() - time_start), 3)
            if verbose:
                print(f"Time elapsed for epoch {epoch + 1}: {time_elapsed}s\n")
            # Last Epoch --> print Error
            if epoch == epochs - 1:
                print(f"Training of final model completed.\n"
                      f"NN Architecture: Layers: {len(self.layer_list)}, "
                      f"Units x layer: {len(self.layer_list[1].net)}")
                total_time_elapsed = round((time.clock() - total_time), 3)
                print(f"Total time elapsed: {total_time_elapsed}s")

    def make_prediction(self, test_set):
        first = True
        for i in range(len(test_set)):
            bound = len(test_set[i])
            test_in = test_set[i][1:bound - 2]
            self.feedforward(test_in, "cup")
            out1 = self.layer_list[-1].out[0]
            out2 = self.layer_list[-1].out[1]
            if first:
                first = False
                outputs = [[out1, out2]]
            else:
                outputs = np.append(outputs, [[out1, out2]], axis=0)
        final_result = np.concatenate((test_set, outputs), axis=1)
        final_result = np.asarray(final_result)
        np.savetxt("results.csv", final_result, delimiter=",", fmt='%s')


def mean_squared_error(target, output, derivative):
    if derivative:
        return np.subtract(target, output)
    else:
        res = np.subtract(target, output) ** 2
        res = np.sum(res, axis=0)
        return res
        # return res / len(output)


def mean_euclidean_error(target_value, neurons_out, derivative):
    if derivative:
        err = mean_euclidean_error(target_value, neurons_out, derivative=False)
        return np.subtract(target_value, neurons_out) * (1 / err)
    res = np.subtract(neurons_out, target_value) ** 2
    res = np.sum(res, axis=0)
    res = np.sqrt(res)
    return res
    # return res / len(neurons_out)


def init_manger(n_layer, n_neurons_layer, n_input, neurons_out_layer, error_func, act_hidden, act_output):
    nn = NeuralNet(error_func)
    nn.init_input_layer(n_input)
    n_neurons_layer_prev = n_input
    for i in range(n_layer):
        nn.init_layer(n_neurons_layer, n_neurons_layer_prev, act_hidden)
        n_neurons_layer_prev = n_neurons_layer
    nn.init_layer(neurons_out_layer, n_neurons_layer_prev, act_output)
    nn.set_weights_pre_training()
    return nn
