import numpy as np
from Layer import Layer
from functions import sigmoid, sigmoid_derivative
from functions import graph

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
            layer.weighted_sum = np.dot(current_inputs, layer.weights)
            
            # Activation function (Sigmoid)
            output = sigmoid(layer.weighted_sum)

            # Store the output of the layer
            layer.output = output

            # Update the current inputs for the next layer
            current_inputs = layer.output
        # Return the output of the last layer
        return current_inputs
    
    def backpropagation(self, y):
        output_layer = self.layers[len(self.layers) - 1]

        # Delta[j] = g'(input[j]) * (yj - aj)
        output_layer.delta = sigmoid_derivative(output_layer.weighted_sum) * (y - output_layer.output)

        # If I don't put these comments I get confused.

        # It was already calculated the delta for the output layer, now we need to calculate the deltas for the hidden layers
        # so if we have 4 layers, the len is 4, but the index for the ouput layer is 3
        # that means the index for the hidden layer is 2, thats why len - 2.
        # The -1 we are iterating through 0, and the other -1 menas that we are doing it in reverse order
        for i in range(len(self.layers) - 2, -1, -1):
            layer = self.layers[i] # Current layer
            next_layer = self.layers[i + 1] # The next layer

            # Delta[i] = g'(input[i]) * (w[ij] * delta[j])
            layer.delta = sigmoid_derivative(layer.weighted_sum) * np.dot(next_layer.delta, next_layer.weights.T)

    def update_weights(self):
        for layer in self.layers:
            # w_ij = w_ij + learning_rate * (ai * delta[j]) 
            layer.weights += self.learning_rate * np.matmul(np.atleast_2d(layer.input).T, np.atleast_2d(layer.delta))

    # X: Matrix of inputs
    # Y: Matrix of outputs
    # epochs: Number of iterations
    def train(self, x_train, y_train, epochs, x_test=None, y_test=None):
        errors = []
        for epoch in range(epochs):
            output_train = self.forward(x_train)
            error_train = np.mean((y_train - output_train) ** 2)
            prediction_train = (output_train > 0.5).astype(int)
            accuracy_train = np.mean(prediction_train == y_train)
            errors.append(error_train)

            # Backpropagation and update weights
            self.backpropagation(y_train)
            self.update_weights()

            # Test the model
            if x_test is not None or y_test is not None:
                output_test = self.forward(x_test)
                error_test = np.mean((y_test - output_test) ** 2)
                prediction_test = (output_test > 0.5).astype(int)
                accuracy_test = np.mean(prediction_test == y_test)
                print(f"Época {epoch + 1}/{epochs}")
                print(f"Error de entrenamiento: {error_train * 100:.2f}%, Precisión de entrenamiento: {accuracy_train * 100:.2f}%")
                print(f"Error de prueba: {error_test * 100:.2f}%, Precisión de prueba: {accuracy_test * 100:.2f}%")
            else:
                print(f"Época {epoch + 1}/{epochs}")
                print(f"Error de entrenamiento: {error_train * 100:.2f}%, Precisión de entrenamiento: {accuracy_train * 100:.2f}%")

            print("-" * 50)

        print("# Entrenamiento completado #")
        print("¿Desea ver un gráfico del proceso de entrenamiento? (s/n)")
        option = input("Opción: ")
        if option.lower() == "s":
            graph(epochs, errors)
        else:
            print("No se mostrará el gráfico.")
        return errors

    def predict(self, X):
        if (X.ndim == 1):
            return self.forward(X)
        else:
            return np.array([self.forward(x) for x in X])
        
    def save_model(self):
        # Save the model to a file
        with open("nn.pkl", 'wb') as f:
            np.save(f, self.layers)

    def load_model(self, filename):
        # Load the model from a file
        with open(filename, 'rb') as f:
            self.layers = np.load(f, allow_pickle=True)


