import sys
import numpy as np
import matplotlib
import time


np.random.seed(0)


class LayerDense:
    # n_inputs is typically the number of values in the lowest level vector, or the number of features in a sample
    # n_neurons is the number of how many neurons you want to create with this layer
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights to be inputs x neurons so in the forwards pass, no transpose is needed
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        # Initialize biases to be a vector of zeros to better catch network errors
        self.biases = np.zeros((1, n_neurons))

    def forwards(self, inputs):
        # Calculate the output of the layer by multiplying the inputs with weights and adding biases
        self.output = np.dot(inputs, self.weights) + self.biases


class ActivationReLU:
    def forwards(self, inputs):
        # Apply the Rectified Linear Unit (ReLU) activation function to the inputs
        # ReLU outputs the input as-is if it's positive, otherwise, it outputs zero
        self.output = np.maximum(0.0, inputs)


class ActivationStep:
    def forwards(self, inputs):
        # Apply the step function activation to the inputs
        # Step function outputs 1 for inputs >= 0, and 0 for inputs < 0
        self.output = np.where(inputs >= 0, 1.0, 0.0)


def create_data(num_points_per_class, num_classes):
    # Calculate the total number of data points needed
    num_total_points = num_points_per_class * num_classes

    # Initialize arrays to store inputs (data points) and labels (class indices)
    inputs = np.zeros((num_total_points, 2))
    labels = np.zeros(num_total_points, dtype="uint8")

    # Generate data for each class
    for class_index in range(num_classes):
        # Calculate the start and end indices for the current class
        start_index = num_points_per_class * class_index
        end_index = num_points_per_class * (class_index + 1)

        # Generate points within a circle (radius and theta values)
        radius = np.linspace(0.0, 1.0, num_points_per_class)
        theta = (
            np.linspace(
                class_index * 4.0, (class_index + 1.0) * 4.0, num_points_per_class
            )
            + np.random.randn(num_points_per_class) * 0.2
        )

        # Convert polar coordinates to Cartesian coordinates and store in the inputs array
        inputs[start_index:end_index] = np.c_[
            radius * np.sin(theta * 2.5), radius * np.cos(theta * 2.5)
        ]

        # Assign the class index as the label for the corresponding data points
        labels[start_index:end_index] = class_index

    return inputs, labels


def main():
    # inputs = [
    #     [1.0, 2.0, 3.0, 2.5],
    #     [2.0, 5.0, -1.0, 2.0],
    #     [-1.5, 2.7, 3.3, -0.8],
    # ]
    inputs, output = create_data(100, 3)

    layer1 = LayerDense(len(inputs[1]), 7)
    activation1 = ActivationReLU()
    activation2 = ActivationStep()

    layer1.forwards(inputs)
    print("Layer1: ", layer1.output)

    activation1.forwards(layer1.output)
    print("Activation1: ", activation1.output)

    activation2.forwards(layer1.output)
    print("Activation2: ", activation2.output)


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
