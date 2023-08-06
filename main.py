import sys
import numpy as np
import matplotlib
import time


np.random.seed(0)


class Layer_Dense:
    # n_inputs is typically the number of values in the lowest level vector, or the number of features in a sample
    # n_neurons is the number of how many neurons you want to create with this layer
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights to be inputs x neurons so in the forwards pass, no transpose is needed
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        # Initialize biases to be a vector of zeros to better catch network errors
        self.biases = np.zeros((1, n_neurons))

    def forwards(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


def main():
    inputs = [
        [1.0, 2.0, 3.0, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8],
    ]

    layer1 = Layer_Dense(len(inputs[1]), 7)
    layer1.forwards(inputs)
    output1 = layer1.output
    print(output1)

    layer2 = Layer_Dense(len(output1[1]), 1)
    layer2.forwards(output1)
    output2 = layer2.output
    print(output2)


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
