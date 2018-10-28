from layer import Layer


class NeuralNet:

    def __init__(self):
        self.layer_list = list()
        self.error_list = list()

    def initialize_layer(self, n_neuron, n_neuron_weights):
        self.layer_list.append(Layer(n_neuron, n_neuron_weights))

    # TODO: find out why sometimes gets an array [[1,2,3]] and sometimes directly [1,2,3]
    def feedforward(self, row_input):
        actual_input = row_input
        for i in range(len(self.layer_list)):
            # TODO: TESTING - TO CLEAN UP
            # if i == len(self.layer_list)-1:
                # Need to convert last input to 1 if >0.5
                # layer = self.layer_list[i]
                # layer.compute_squash_layer_final()
                # return
            layer = self.layer_list[i]
            layer.compute_input_layer(actual_input)
            next_input = layer.compute_squash_layer()[0]
            actual_input = next_input

    # Not entirely sure about this error calculation
    def compute_delta_output_layer(self, target):
        error_output_layer = 0.00
        output_layer = self.layer_list[-1]
        for neuron in output_layer.neurons:
            err = neuron.compute_delta_output(target)
            # TODO: CHECK CODE HERE
            error_output_layer += 0.5 * (err ** 2)
        return error_output_layer

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
    def update_weights(self, learning_rate):
        for layer in self.layer_list:
            for neuron in layer.neurons:
                for i in range(len(neuron.weights[0])):
                    new_weight = neuron.weights[0, i] + learning_rate * neuron.delta * neuron.inputs_list[0, i]
                    neuron.weights[0, i] = new_weight

    def training(self, n_epochs, tr_set, learning_rate=0.001, verbose=False, monksDataset=False):
        for j in range(n_epochs):
            print("\nEPOCH {} ___________________________".format(j + 1))
            epoch_error = 0.00
            for i in range(len(tr_set)):
                # if verbose:
                #     print(f"Training_sample {i+1} of {len(tr_set)}")
                tr_in = tr_set[i][:-1]
                target = tr_set[i][-1]
                if monksDataset:
                    tr_in, target = self.preprocess_monk_dataset(tr_set[i])
                self.feedforward(tr_in)
                err_out = self.compute_delta_output_layer(target)
                err_hid = self.compute_delta_hidden_layer()
                epoch_error += err_out
                self.update_weights(learning_rate)
            print(f"Total Error for Epoch: {round(epoch_error, 5)}")
            self.error_list.append((j, epoch_error))

    # for MonkDataset -> remove last column and set
    # features and first column as target desired output
    def preprocess_monk_dataset(self, tr_set_row):
        tr_row = tr_set_row[:-1]
        tr_in = tr_row[1:]
        target = tr_row[0]
        return tr_in, target

