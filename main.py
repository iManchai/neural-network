import os
from tkinter import filedialog
import numpy as np
from NeuralNetwork import NeuralNetwork



# TAREA 2
# PERCEPTRON MULTICAPAS

def main():
    # Initialize the neural network parameters
    try:
      print("¿Cómo desea crear la red neuronal?")
      print("1. Manual")
      print("2. Cargar desde archivo")
      option = input("Seleccione una opción (1/2): ").strip()

      if option == "1":
        # Get the parameters for the neural network
        num_inputs_neurons = int(input("Ingrese el número de neuronas de entrada: "))
        num_hidden_layers = int(input("Ingrese el número de capas ocultas: "))
        num_neurons_hidden_layer = int(input("Ingrese el número de neuronas por capa oculta: "))
        num_output_neurons = int(input("Ingrese el número de neuronas de salida: "))

        # Load the training data file
        x_train_file = input("Ingrese el nombre del archivo de entrenamiento (por ejemplo: X_train.txt): ")
        x_train_path = os.path.join(os.path.dirname(__file__), x_train_file)
        if not os.path.isfile(x_train_path):
            print(f"Error: El archivo '{x_train_file}' no existe en la carpeta actual.")
            return

        try:
          with open(x_train_path, "r") as f:
            X_train = []
            for line in f:
                vector = [float(x.strip()) for x in line.strip().split(",") if x.strip()]
                X_train.append(vector)
            X_train = np.array(X_train)
        except Exception as e:
            print(f"Error al leer el archivo de entrenamiento: {e}")
            return

        # Load the training labels file
        y_train_file = input("Ingrese el nombre del archivo de etiquetas de entrenamiento (por ejemplo: y_train.txt): ")
        y_train_path = os.path.join(os.path.dirname(__file__), y_train_file)
        if not os.path.isfile(y_train_path):
            print(f"Error: El archivo '{y_train_file}' no existe en la carpeta actual.")
            return

        try:
          with open(y_train_path, "r") as f:
            y_train = []
            for line in f:
                vector = [float(x.strip()) for x in line.strip().split(",") if x.strip()]
                y_train.append(vector)
            y_train = np.array(y_train)
        except Exception as e:
            print(f"Error al leer el archivo de etiquetas de entrenamiento: {e}")
            return

        # Load the test data file
        x_test_file = input("Ingrese el nombre del archivo de prueba (por ejemplo: X_test.txt): ")
        x_test_path = os.path.join(os.path.dirname(__file__), x_test_file)
        if not os.path.isfile(x_test_path):
            print(f"Error: El archivo '{x_test_file}' no existe en la carpeta actual.")
            return

        try:
          with open(x_test_path, "r") as f:
            X_test = []
            for line in f:
                vector = [float(x.strip()) for x in line.strip().split(",") if x.strip()]
                X_test.append(vector)
            X_test = np.array(X_test)
        except Exception as e:
            print(f"Error al leer el archivo de prueba: {e}")
            return

        # Load the test labels file
        y_test_file = input("Ingrese el nombre del archivo de etiquetas de prueba (por ejemplo: y_test.txt): ")
        y_test_path = os.path.join(os.path.dirname(__file__), y_test_file)
        if not os.path.isfile(y_test_path):
            print(f"Error: El archivo '{y_test_file}' no existe en la carpeta actual.")
            return

        try:
          with open(y_test_path, "r") as f:
            y_test = []
            for line in f:
                vector = [float(x.strip()) for x in line.strip().split(",") if x.strip()]
                y_test.append(vector)
            y_test = np.array(y_test)
        except Exception as e:
            print(f"Error al leer el archivo de etiquetas de prueba: {e}")
            return
           
        
        epochs = int(input("Ingrese el número de épocas: "))

        # Create the neural network
        nn = NeuralNetwork(num_inputs_neurons, num_hidden_layers, num_neurons_hidden_layer, num_output_neurons, learning_rate=0.1)

        # Train the neural network
        nn.train(X_train, y_train, X_test, epochs)
        # Test the neural network
        for i in range(len(X_test)):
            output = nn.predict(X_test[i])
            print(f"Input: {X_test[i]}, Predicted Output: {output}, Expected Output: {y_test[i][0]}")

      elif option == "2":
        file_path = input("Ingrese la ruta del archivo de configuración: ")
      else:
        print("Opción no válida. Por favor, ingrese '1' o '2'.")
        return
    except ValueError:
      print("Error: Por favor, ingrese valores numéricos válidos.")
      return
    except FileNotFoundError:
      print("Error: Archivo de configuración no encontrado.")
      return

    
  


    # nn.train(X, y, X_test, epochs)
    # for i in range(len(X)):
    #     output = nn.predict(X[i])
    #     print(f"Input: {X[i]}, Predicted Output: {output}, Expected Output: {y[i][0]}")

main()