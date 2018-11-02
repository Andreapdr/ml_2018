import numpy as np
from layer import Layer
from pprint import pprint


class NeuralNet:

    def __init__(self):
        self.layer_list = list()
        self.error_list = list()
        self.accuracy_list = list()

    def initialize_layer(self, n_neuron, n_neuron_weights):
        self.layer_list.append(Layer(n_neuron, n_neuron_weights))

    # TODO: find out why sometimes gets an array [[1,2,3]] and sometimes directly [1,2,3]
    def feedforward(self, row_input):
        actual_input = row_input
        for i in range(len(self.layer_list)):
            layer = self.layer_list[i]
            layer.compute_input_layer(actual_input)
            if i == len(self.layer_list)-1:
                layer.compute_squash_layer()
                return
            # TODO: FIND OUY WHY I HAVE TO ADD [0]
            next_input = layer.compute_squash_layer()[0]
            actual_input = next_input

    def compute_output(self, nn_input):
        for i in range(len(self.layer_list)):
            layer = self.layer_list[i]
            layer.compute_input_layer(nn_input)

    def compute_delta_output_layer(self, target):
        error_output_layer = 0.00
        output_layer = self.layer_list[-1]
        for j, neuron in enumerate(output_layer.neurons):
            # for multiple target values
            # err = neuron.compute_delta_output(target[j])
            err = neuron.compute_delta_output(target)
            # TODO: CHECK CODE HERE
            error_output_layer += 0.5 * (err ** 2)
        return error_output_layer

    # iterate the layer_list in reverse order starting
    # j is equal to the index of neuron-> retrieve its weights
    def compute_delta_hidden_layer(self):
        error_input_layer = 0.00
        for i in range(len(self.layer_list)-1, 0, -1):
            layer = self.layer_list[i-1]
            next_layer = self.layer_list[i]
            for j, neuron in enumerate(layer.neurons):
                err = neuron.compute_delta_hidden(next_layer, j)
                error_input_layer += err
        return error_input_layer

    # Updating weights: w_new += delta_Wji
    # where delta_Wji = lr * delta_j * input_ji
    def update_weights(self, learning_rate, momentum, alpha):
        previous_weight_update = 0.0
        for layer in self.layer_list:
            for neuron in layer.neurons:
                for i in range(len(neuron.weights[0])):
                    # TODO: CHECK MOMENTUM
                    # TODO: IMPLEMENT WEIGHT DECAY TIKHONOV
                    momentum_term = momentum * previous_weight_update
                    weight_update = momentum_term + learning_rate * neuron.delta * neuron.inputs_list[0, i]
                    previous_weight_update = weight_update
                    new_weight = neuron.weights[0, i] + weight_update - (2 * alpha * neuron.weights[0, i])
                    neuron.weights[0, i] = new_weight

    def training(self, n_epochs, tr_set, learning_rate=0.01, momentum=0.00, alpha=0.00,
                 step_decay=False, lr_decay=False, verbose=False):
        current_learning_rate = learning_rate
        for j in range(n_epochs):
            print("\nEPOCH {} ___________________________".format(j + 1))
            epoch_error = 0.00
            epoch_accuracy = 0.00
            # use different order of patterns in different epochs
            np.random.shuffle(tr_set)
            # STEP DECAY: learning rate is cut by 10 every 10 epochs
            if step_decay:
                if j % 10 == 0 and j != 0:
                    current_learning_rate = current_learning_rate/10
            if lr_decay:
                current_learning_rate -= 0.001
            # For every sample in dataset:
            for i in range(len(tr_set)):
                # First elem of set == target
                tr_in = tr_set[i][1:]
                target = tr_set[i][0]
                self.feedforward(tr_in)
                err_out = self.compute_delta_output_layer(target)
                self.compute_delta_hidden_layer()
                epoch_error += err_out
                self.update_weights(learning_rate, momentum, alpha)
                nn_output = self.layer_list[1].neurons[0].compute_output_final()
                epoch_accuracy += 1 - abs(target - nn_output)
                if verbose:
                    print(f"Training_sample {i+1} of {len(tr_set)}")
            print(f"Total Error for Epoch: {round(epoch_error/len(tr_set), 5)}")
            # Compute normalization of squared error --> ( epoch_error/len(tr(set)))
            self.error_list.append((j, epoch_error/len(tr_set)))
            self.accuracy_list.append((j, epoch_accuracy/len(tr_set)))
        if verbose:
            print(f"Final NN: Weights:")
            for layer in self.layer_list:
                for neuron in layer.neurons:
                    pprint(neuron.weights)

    def test(self, test_set):
        print("\nTEST RESULTS___________________________________________")
        correct_predictions = 0
        for i in range(len(test_set)):
            test_in = test_set[i][1:]
            target = test_set[i][0]
            self.feedforward(test_in)
            nn_output = self.layer_list[1].neurons[0].compute_output_final()
            correct_predictions += 1 - abs(target - nn_output)
        print(f"\nAccuracy: {correct_predictions/len(test_set)}\n"
              f"Total Predictions: {len(test_set)}")


