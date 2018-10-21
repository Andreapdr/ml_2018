from layer import Layer


class NeuralNet:

    def __init__(self):
        self.layer_list = list()

    def initialize_layer(self, n_neuron, n_neuron_weights):
        self.layer_list.append(Layer(n_neuron, n_neuron_weights))

    def feedforward(self, row_input):                   # TODO: row_input is placeholder. To implement for a matrix of inputs vector
        actual_input = row_input
        for j in range(len(row_input)):                 # for every row of training_set (== for every instance sample)
            for i in range(len(self.layer_list)):       # for every layer starting from input
                layer = self.layer_list[i]
                # for INPUT LAYER (i == 0)
                layer.compute_input_layer(actual_input)     # compute the layer input for every node (w_nodej * input_x)
                next_input = layer.compute_squash_layer()   # compute sigmoid squashing and create from the values an
                actual_input = next_input                   # input vector for the next layer. Last actual_input is
                                                            # just the output vector (=results)

    def feedforward2(self, row_input):                  # previous function was cycling 2 times over the dataset -
        actual_input = row_input                        # useful only for testing a single feedforward
        for i in range(len(self.layer_list)):
            layer = self.layer_list[i]
            layer.compute_input_layer(actual_input)
            next_input = layer.compute_squash_layer()
            actual_input = next_input

    def compute_delta_output_layer(self, target):
        output_layer = self.layer_list[-1]
        for neuron in output_layer.neurons:
            neuron.compute_delta_output(target)
            # print(neuron.compute_delta_output(target))

    def compute_delta_hidden_layer(self):
        for i in range(len(self.layer_list)-1, 0, -1):      # iterate the layer_list in reverse order starting
            # print(f"\nevaluating_delta for layer {i}")
            layer = self.layer_list[i-1]                    # TODO: Check this iteration 'cause i'm not really sure about it
            next_layer = self.layer_list[i]
            for j, neuron in enumerate(layer.neurons):
                neuron.compute_delta_hidden(next_layer, j)  # j is equal to the index of neuron-> retrieve its weights

    def update_weights(self, learning_rate=0.001):          # Updating weights: w_new += delta_Wji
        for layer in self.layer_list:                       # where delta_Wji = lr * delta_j * input_ji
            for neuron in layer.neurons:
                for i in range(len(neuron.weights[0])):
                    new_weight = neuron.weights[0, i] * learning_rate * neuron.delta * neuron.network_in  # TODO: check if neuron.network_in is actually x_ji
                    neuron.weights[0, 1] = new_weight

    def training(self, n_epochs, tr_set, verbose=False):
        for j in range(n_epochs):
            print("\nEPOCH {} ___________________________".format(j + 1))
            for i in range(len(tr_set)):
                if verbose:
                    print(f"Training_sample {i+1} of {len(tr_set)}")
                tr_in, target = self.preprocess_monk_dataset(tr_set[i])
                self.feedforward2(tr_in)
                self.compute_delta_output_layer(target)
                self.compute_delta_hidden_layer()
                self.update_weights()

    def preprocess_monk_dataset(self, tr_set_row):
        tr_row = tr_set_row[:-1]                        # for MonkDataset -> remove last column and set
        tr_in = tr_row[1:]                              # features and first column as target desired output
        target = tr_row[0]
        return tr_in, target

