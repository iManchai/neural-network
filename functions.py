import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Auxiliar Functions

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Load the CSV file for general data
def load_file(file=None):
  path = os.path.join(os.path.dirname(__file__), file)

  try:
    data = np.genfromtxt(path, delimiter=',')
    if data.ndim == 1:
      # If the data is one-dimensional, reshape it to be two-dimensional (column vector)
      data = data.reshape(-1, 1)
  except Exception as e:
      print(f"Error al leer el archivo de entrenamiento: {e}")
      sys.exit(1)
  return data

def graph(epochs, errors):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), [err * 100 for err in errors], label="Error %", marker='o')
    plt.xlabel("Época")
    plt.ylabel("Error (%)")
    plt.title("Error por Época")
    plt.legend()
    plt.grid(True)
    plt.show()