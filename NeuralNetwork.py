import numpy as np
from Layer import Layer
from functions import sigmoid, sigmoid_derivative

class NeuralNetwork:
    def __init__(self, num_inputs_neurons, num_hidden_layers, num_neurons_hidden_layer, num_output_neurons, learning_rate=0.01):
      # Inicialize the neural network
        self.learning_rate = learning_rate
        self.layers = []

        for i in range(num_hidden_layers + 1):
            # First Hidden Layer
            if i == 0:
                self.layers.append(Layer(num_inputs_neurons, num_neurons_hidden_layer))
            # Hidden Layers
            elif 0 < i < num_hidden_layers:
                self.layers.append(Layer(num_neurons_hidden_layer, num_neurons_hidden_layer))
            # Output Layer
            else:
                self.layers.append(Layer(num_neurons_hidden_layer, num_output_neurons))


    def forward(self, x):
        # Initialize the first inputs (1st time for the input layer)
        current_inputs = x
        for layer in self.layers:
            # Take the current inputs and pass them through the layer
            layer.input = current_inputs

            # Weighted sum
            layer.sum = np.dot(layer.input, layer.weights)
            
            # Activation function (Sigmoid)
            output = sigmoid(layer.sum)

            # Store the output of the layer
            layer.output = output

            # Update the current inputs for the next layer
            current_inputs = layer.output
        return current_inputs
    
    def backpropagation(self, y):
        output_layer = self.layers[-1]

        # Delta[j] = g'(input[j]) * (yj - aj)
        output_layer.delta = sigmoid_derivative(output_layer.output) * (y - output_layer.output)

        # It was already calculated the delta for the output layer, now we need to calculate the deltas for the hidden layers
        # so if we have 4 layers, the len is 4, but the index for the ouput layer is 3
        # that means the index for the hidden layer is 2, thats why len - 2.
        # The -1 we are iterating through 0, and the other -1 menas that we are doing it in reverse order
        for i in range(len(self.layers) - 2, -1, -1):
            layer = self.layers[i] # Current layer
            next_layer = self.layers[i + 1] # The nex layer

            # Delta[i] = g'(input[i]) * (delta[j] * w[j][i])
            layer.delta = sigmoid_derivative(layer.output) * np.dot(next_layer.weights, next_layer.delta)
