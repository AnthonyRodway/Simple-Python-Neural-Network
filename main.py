import sys
import numpy as np
import matplotlib
import time


def main():
    inputs = [1, 2, 3, 2.5]
    weights = [
        [0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87],
    ]
    biases = [2, 3, 0.5]

    output = np.dot(weights, inputs) + biases

    print(output)


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
