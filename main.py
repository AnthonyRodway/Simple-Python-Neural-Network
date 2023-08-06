import sys
import numpy as np
import matplotlib
import time


def main():
    inputs = [
        [1.0, 2.0, 3.0, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8],
    ]

    weights = [
        [0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87],
    ]
    biases = [2.0, 3.0, 0.5]

    weights2 = [
        [0.1, 0.2, -0.3],
        [0.6, -0.5, 0.6],
        [-0.1, -0.9, 0.7],
    ]
    biases2 = [-1.0, 1.0, -0.5]

    layer1_outputs = np.dot(inputs, np.transpose(weights)) + biases

    print("Layer1: ", layer1_outputs)

    layer2_outputs = np.dot(layer1_outputs, np.transpose(weights2)) + biases2

    print("Layer2: ", layer2_outputs)


if __name__ == "__main__":
    # Library and Python versions
    # print("Python: ", sys.version)
    # print("Numpy: ", np.__version__)
    # print("Matplotlib: ", matplotlib.__version__)

    # Get the start time of the program
    start = time.time()

    # Call the main file function
    main()

    # Get the end time of the program
    end = time.time()

    # Calculate the elapsed time
    elapsed = end - start
    print(f"Elapsed time: {elapsed:.3f} seconds")
