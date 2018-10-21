import numpy as np
import math

row_in = np.random.rand(1,2)
out_y = np.array([0])

# testing 2 * 3 * 2 network stochastic fully connected

# hidden layer: 3 nodes with 2 weights each
# every neuron should have a val_in and val_out (sigmoid function applied)
# to compute val_in: dot product row_in, neuron's weights


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

    def compute_delta_output_layer(self, target):
        output_layer = self.layer_list[-1]
        for neuron in output_layer.neurons:
            neuron.compute_delta_output(target)
            # print(neuron.compute_delta_output(target))

    def compute_delta_hidden_layer(self):
        for i in range(len(self.layer_list)-1, 0, -1):      # iterate the layer_list in reverse order starting
            print(f"evaluating_delta for layer {i}")
            layer = self.layer_list[i-1]                    # TODO: Check this iteration 'cause i'm not really sure about it
            next_layer = self.layer_list[i]
            for j, neuron in enumerate(layer.neurons):
                # print(f"enumerating neuron: {j}")
                neuron.compute_delta_hidden(next_layer, j)    # j is equal to the index of neuron-> retrieve its weights


class Layer:

    def __init__(self, n_neurons, n_neurons_weights):
        self.neurons = list()
        self.out_vector = np.empty([1, n_neurons])
        for i in range(n_neurons):
            self.neurons.append(Neuron(n_neurons_weights))

    def compute_input_layer(self, row_input):
        for i in range(len(self.neurons)):
            self.neurons[i].compute_network_in(row_input)

    def compute_squash_layer(self):
        for i in range(len(self.neurons)):
            r = self.neurons[i].compute_output()
            self.out_vector[0, i] = r
        return self.out_vector


class Neuron:

    def __init__(self, n_weights):
        self.weights = np.random.rand(1, n_weights)
        self.network_in = 0.00
        self.output = 0.00
        self.delta_out = 0.00
        self.delta_hidden = 0.00

    def compute_network_in(self, row_inputs):
        self.network_in = np.dot(row_inputs, self.weights.T)        # return an ndarray object !
        return self.network_in

    def get_weight(self):
        return self.weights()

    def compute_output(self):
        self.output = sigmoid_function(self.network_in)
        return self.output

    def compute_delta_output(self, target):
        derivative_activation_function = self.output * (1 - self.output)          # for sigmoidal activation function
        self.delta_out = (target - self.output) * derivative_activation_function
        return self.delta_out

    def compute_delta_hidden(self, next_layer, index_of_nueron_prev_layer):
        derivative_activation_function = self.output * (1 - self.output)          # for sigmoidal activation function
        hidden_error = 0.00
        for neuron in next_layer.neurons:
            hidden_error += neuron.delta_out * neuron.weights[0, index_of_nueron_prev_layer]    # hidden_error = Delta next_layer neuron
        print(f"For Neuron: hidden error: {hidden_error}")                                      # multplied by weight[j] next_layer neuron
        delta_hidden = hidden_error * derivative_activation_function
        # print(delta_hidden)
        self.delta_hidden = delta_hidden
        return delta_hidden


# the weight matrix at level t could be implemented as a matrix of weights.


def sigmoid_function(x):
    return 1 / (1 + math.exp(-x))


if __name__ == "__main__":
    # structure testing
    nn = NeuralNet()  # initialize empty network = list containing layers
    nn.initialize_layer(3, 2)  # set first layer (3 neuron, 2 weights each)
    nn.initialize_layer(2, 3)  # set out_layer (2 neuron, 3 weights each)

    nn.feedforward(row_in)
    nn.compute_delta_output_layer(out_y)
    nn.compute_delta_hidden_layer()
