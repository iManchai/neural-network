import os
from load_files import load_file
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

        # Load the CSV file for training data
        x_train_file = input("Ingrese el nombre del archivo CSV de entrenamiento (por ejemplo: X_train.csv): ")
        X_train = load_file(x_train_file)

        # Load the CSV file for training labels
        y_train_file = input("Ingrese el nombre del archivo CSV de etiquetas de entrenamiento (por ejemplo: y_train.csv): ")
        y_train = load_file(y_train_file)

        # Load the CSV file for test data
        x_test_file = input("Ingrese el nombre del archivo CSV de prueba (por ejemplo: X_test.csv): ")
        X_test = load_file(x_test_file)

        # Load the CSV file for test labels
        y_test_file = input("Ingrese el nombre del archivo CSV de etiquetas de prueba (por ejemplo: y_test.csv): ")
        y_test = load_file(y_test_file)

        epochs = int(input("Ingrese el número de épocas: "))

        # Create the NN
        nn = NeuralNetwork(num_inputs_neurons, num_hidden_layers, num_neurons_hidden_layer, num_output_neurons, learning_rate=0.1)

        # Train the NN
        nn.train(X_train, y_train, X_test, y_test, epochs)

        # Test the NN    
        for i in range(len(X_test)):
            output = nn.predict(X_test[i])
            print(f"Input: {X_test[i]}, Predicted Output: {output}, Expected Output: {y_test[i]}")

        try:
          print("Que desea hacer con el modelo entrenado?")
          print("1. Ejecutar Feed Forward")
          print("2. Seguir entrenando la red")
          print("3. Guardar el modelo")
          print("4. Ver gráfica del error")
          option = input("Seleccione una opción (1/2/3/4): ")

          if option == "1":
            option_feed_forward = input("Desea ingresar el vector manualmente o cargarlo desde un archivo? (1/2): ")
            if option_feed_forward == "1":
              vector = input("Ingrese el vector de entrada (separado por comas): ")
              vector = np.array([float(i) for i in vector.split(",")])
              if vector.ndim == 1:
                # Transform the vector to a 2D array row vector
                vector = vector.reshape(1, -1)
              output = nn.predict(vector)
              print(f"Input: {vector}, Predicted Output: {output}")

            elif option_feed_forward == "2":
              vector_file = input("Ingrese el nombre del archivo CSV de entrada (por ejemplo: vector.csv): ")
              vector = load_file(vector_file)

              for i in range(len(vector)):
                if vector[i].ndim == 1:
                  vector[i] = vector[i].reshape(1, -1)
                output = nn.predict(vector[i])
                print(f"Input: {vector[i]}, Predicted Output: {output}")
            

        except ValueError as e:
          print("Error: Por favor, ingrese valores numéricos válidos.", e)
          return

      elif option == "2":
        file_path = input("Ingrese la ruta del archivo de configuración: ")
      else:
        print("Opción no válida. Por favor, ingrese '1' o '2'.")
        return
    except ValueError as e:
      print("Error: Por favor, ingrese valores numéricos válidos.", e)
      return
    except FileNotFoundError:
      print("Error: Archivo de configuración no encontrado.")
      return

main()