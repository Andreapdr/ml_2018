import numpy as np
from layer1_2 import Layer


class NeuralNet:

    def __init__(self):
        self.layer_list = list()
        self.tr_error_list = list()
        self.val_error_list = list()

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
            next_input = layer.compute_squash_layer()[0]
            actual_input = next_input

    def compute_output(self, nn_input):
        for i in range(len(self.layer_list)):
            layer = self.layer_list[i]
            layer.compute_input_layer(nn_input)

    # Not entirely sure about this error calculation
    def compute_delta_output_layer(self, target):
        error_output_layer = 0.00
        output_layer = self.layer_list[-1]
        for neuron in output_layer.neurons:
            err = neuron.compute_delta_output(target)
            # TODO: CHECK CODE HERE
            error_output_layer += 0.5 * (err ** 2)
        return 0.5 * error_output_layer

    # TODO: Check this iteration 'cause i'm not really sure about it
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
    def update_weights(self, learning_rate, momentum):
        previous_weight_update = 0.0
        for layer in self.layer_list:
            for neuron in layer.neurons:
                for i in range(len(neuron.weights[0])):
                    momentum_term = momentum * previous_weight_update
                    weight_update = momentum_term + learning_rate * neuron.delta * neuron.inputs_list[0, i]
                    previous_weight_update = weight_update
                    new_weight = neuron.weights[0, i] + weight_update
                    neuron.weights[0, i] = new_weight

    def training(self, n_epochs, tr_set, val_set, learning_rate=0.01, momentum=0.05, verbose=False, monksDataset=False):
        current_learning_rate =learning_rate
        for j in range(n_epochs):
            correct_predictions = 0
            print("\nEPOCH {} ___________________________".format(j + 1))
            # use different order of patterns in different epochs
            np.random.shuffle(tr_set)
            # learning rate decreases through epochs
            current_learning_rate -= 0.001
            for i in range(len(tr_set)):
                # if verbose:
                #     print(f"Training_sample {i+1} of {len(tr_set)}")
                tr_in = tr_set[i][:-1]
                target = tr_set[i][-1]
                if monksDataset:
                    tr_in, target = self.preprocess_monk_dataset(tr_set[i])
                self.feedforward(tr_in)
                self.compute_delta_output_layer(target)
                self.compute_delta_hidden_layer()
                nn_output = self.layer_list[1].neurons[0].compute_output_final()
                correct_predictions += 1 - abs(target - nn_output)
                self.update_weights(learning_rate, momentum)
            epoch_error = 1.0 - (correct_predictions / len(tr_set))
            print(f"Total Error for Epoch: {round(epoch_error, 5)}")
            self.tr_error_list.append((j, epoch_error))
            val_error = self.test(val_set, monksDataset=True, verbose=False)
            self.val_error_list.append((j, val_error))
            if round(epoch_error, 5) == 0.0:
                break
        print(f"Final NN: Weights:")
        for layer in self.layer_list:
            for neuron in layer.neurons:
                print(neuron.weights)

    def test(self, test_set, monksDataset=False, verbose=False):
        if verbose:
            print("\nTest results:______________________")
        correct_predictions = 0
        for i in range(len(test_set)):
            test_in = test_set[i][:-1]
            target = test_set[i][-1]
            if monksDataset:
                test_in, target = self.preprocess_monk_dataset(test_set[i])
            self.feedforward(test_in)
            nn_output = self.layer_list[1].neurons[0].compute_output_final()
            correct_predictions += 1 - abs(target - nn_output)
        if verbose:
            print("Accuracy:")
            print(correct_predictions/len(test_set))
        return 1.0 - (correct_predictions/len(test_set))


    # for MonkDataset -> remove last column and set
    # features and first column as target desired output
    @staticmethod
    def preprocess_monk_dataset(tr_set_row):
        tr_row = tr_set_row[:-1]
        tr_in = tr_row[1:]
        target = tr_row[0]
        return tr_in, target