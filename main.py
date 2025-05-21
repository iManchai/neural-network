import numpy as np
from NeuralNetwork import NeuralNetwork



# TAREA 2
# PERCEPTRON MULTICAPAS

def main():
    # Inicializar la red neuronal
    try:
      opcion = input("¿Desea crear la red neuronal manualmente o cargarla desde un archivo? (manual/cargar): ").strip().lower()
      if opcion == "manual":
        num_inputs_neurons = int(input("Ingrese el número de neuronas de entrada: "))
        num_hidden_layers = int(input("Ingrese el número de capas ocultas: "))
        num_neurons_hidden_layer = int(input("Ingrese el número de neuronas por capa oculta: "))
        num_output_neurons = int(input("Ingrese el número de neuronas de salida: "))
        epochs = int(input("Ingrese el número de épocas: "))
      elif opcion == "cargar":
        file_path = input("Ingrese la ruta del archivo de configuración: ")
        with open(file_path, "r") as file:
          # Se espera que el archivo contenga los parámetros en líneas separadas:
          # línea 1: num_inputs_neurons
          # línea 2: num_hidden_layers
          # línea 3: num_neurons_hidden_layer
          # línea 4: num_output_neurons
          # línea 5: epochs
          lines = file.readlines()
          if len(lines) < 5:
            print("Error: el archivo no contiene todos los parámetros requeridos.")
            return
          num_inputs_neurons = int(lines[0].strip())
          num_hidden_layers = int(lines[1].strip())
          num_neurons_hidden_layer = int(lines[2].strip())
          num_output_neurons = int(lines[3].strip())
          epochs = int(lines[4].strip())
      else:
        print("Opción no válida. Por favor, ingrese 'manual' o 'cargar'.")
        return
    except ValueError:
      print("Error: Por favor, ingrese valores numéricos válidos.")
      return
    except FileNotFoundError:
      print("Error: Archivo de configuración no encontrado.")
      return

    # Create the neural network
    nn = NeuralNetwork(num_inputs_neurons, num_hidden_layers, num_neurons_hidden_layer, num_output_neurons, learning_rate=0.1)

    
  


    nn.train(X, y, X_test, epochs)
    for i in range(len(X)):
        output = nn.predict(X[i])
        print(f"Input: {X[i]}, Predicted Output: {output}, Expected Output: {y[i][0]}")

main()