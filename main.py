
import os
import sys
import numpy as np
from functions import load_file
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
        nn.train(X_train, y_train, epochs, X_test, y_test,)

        # Test the NN    
        for i in range(len(X_test)):
            output = nn.predict(X_test[i])
            print(f"Input: {X_test[i]}, Predicted Output: {output}, Expected Output: {y_test[i]}")

      elif option == "2":
        nn_file_name = input("Ingrese el nombre del archivo de configuración: ")
        path = os.path.join(os.path.dirname(__file__), nn_file_name)
        if os.path.exists(path):
          # Load the model from the file
          # Create the NN (No matter the params, they will be overwritten as the layers and its weights will be loaded)
          nn = NeuralNetwork(2, 4, 4, 2, learning_rate=0.1)
          nn.load_model(nn_file_name)
          print("Modelo cargado exitosamente.")
        else:
          print("Error: El archivo de configuración no existe.")
          return
      else:
        print("Opción no válida. Por favor, ingrese '1' o '2'.")
        return
      
      # ============================================================================ #
      # What to do with the trained model
      # After training or loading the model, we can ask the user what to do with it
      while True:
        print("Que desea hacer con el modelo entrenado?")
        print("1. Ejecutar Feed Forward")
        print("2. Seguir entrenando la red")
        print("3. Guardar el modelo")
        print("4. Salir")
        option = input("Seleccione una opción (1/2/3/4): ")

        if option == "1":
          option_feed_forward = input("Desea ingresar el vector manualmente o cargarlo desde un archivo? (1/2): ")

          if option_feed_forward == "1":
            vector = input("Ingrese el vector de entrada (E.g:1,1,3): ")
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
        elif option == "2":
          # Ask for the new training data
          new_train_file = input("Ingrese el nombre del nuevo archivo CSV de entrenamiento (por ejemplo: X_train.csv): ")
          new_X_train = load_file(new_train_file)
          new_train_y_file = input("Ingrese el nombre del nuevo archivo CSV de etiquetas de entrenamiento (por ejemplo: y_train.csv): ")
          new_y_train = load_file(new_train_y_file)
          new_epochs = int(input("Ingrese el número de épocas para seguir entrenando: "))

          # Continue training the NN
          nn.train(new_X_train, new_y_train, new_epochs)
        elif option == "3":
          # Ask for the file name to save the model
          nn.save_model()
          print(f"Modelo guardado en nn.pkl")
        elif option == "4": 
          sys.exit("Saliendo del programa.")
        else:
          print("Opción no válida. Por favor, ingrese '1', '2', '3, o '4'.")
    
    except ValueError as e:
      print("Error: Por favor, ingrese valores numéricos válidos.", e)
      return
    except FileNotFoundError:
      print("Error: Archivo de configuración no encontrado.")
      return

if __name__ == "__main__":
  main()