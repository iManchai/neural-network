import os
import sys

import numpy as np

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