import numpy as np

class Layer:
    def __init__(self, num_inputs, num_neurons):
        # Initialize the layer with random weights
        self.weights = np.random.randn(num_inputs, num_neurons)
        
        # Input for the layer
        self.input = None
        # Output of the layer
        self.output = None
        # Delta for backpropagation
        self.delta = None
        # Weighted sum
        self.weighted_sum = None