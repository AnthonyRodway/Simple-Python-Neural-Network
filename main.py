import sys
import numpy as np
import matplotlib


def main():
    inputs = [1.2, 5.1, 2.1]
    weights = [3.1, 2.3, 4.5]
    bias = 3

    output = np.dot(inputs, weights) + bias
    print(output)


if __name__ == "__main__":
    # Library and Python versions
    # print("Python: ", sys.version)
    # print("Numpy: ", np.__version__)
    # print("Matplotlib: ", matplotlib.__version__)

    # Call the main file function
    main()
