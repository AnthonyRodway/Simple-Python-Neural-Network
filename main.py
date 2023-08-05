import sys
import numpy as np
import matplotlib


def main():
    inputs = [1, 2, 3, 2.5]
    weights1 = [0.2, 0.8, -0.5, 1.0]
    weights2 = [0.5, -0.91, 0.26, -0.5]
    weights3 = [-0.26, -0.27, 0.17, 0.87]
    bias1 = 2
    bias2 = 3
    bias3 = 0.5

    output = [
        np.dot(inputs, weights1) + bias1,
        np.dot(inputs, weights2) + bias2,
        np.dot(inputs, weights3) + bias3,
    ]
    print(output)


if __name__ == "__main__":
    # Library and Python versions
    # print("Python: ", sys.version)
    # print("Numpy: ", np.__version__)
    # print("Matplotlib: ", matplotlib.__version__)

    # Call the main file function
    main()
