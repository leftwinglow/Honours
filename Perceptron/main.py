import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data():
    URL_ = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    data = pd.read_csv(URL_, header=None)

    # make the dataset linearly separable
    data = data[:100]
    data[4] = np.where(data.iloc[:, -1] == "Iris-setosa", 0, 1)
    data = np.asmatrix(data, dtype="float64")
    return data


data = load_data()


import numpy as np


# Define the step function used as the activation function for the perceptron
def step(x):
    """Activation function: returns 1 if x is positive, 0 otherwise."""
    return 1 if x > 0 else 0


# Define the perceptron function
def perceptron(data: np.matrix, epochs: int = 5):
    """
    Implements a simple perceptron model.

    Parameters:
    - data: A numpy matrix where each row is an example. The last column is assumed to be the label.
    - epochs: The number of passes through the entire dataset the algorithm will make.

    Returns:
    A list containing the number of misclassified examples in each epoch.
    """

    # Separate the features from the labels in the input data
    features = data[:, :-1]  # All rows, all but the last column
    labels = data[:, -1]  # All rows, only the last column

    # Initialize the weights and bias to random values
    # Weights are randomly initialized for each feature, and bias is initialized as a single value
    # Note: Simple perceptrons usually initialsie weights to 0.
    w = np.random.rand(1, features.shape[1])
    b = np.random.rand(1)

    # Prepare a list to keep track of the number of misclassifications in each epoch
    results = []

    # Loop through the dataset 'epochs' times
    for _ in range(epochs):
        misclassified = 0  # Count of misclassified examples, reset each epoch

        # Iterate over each example in the dataset
        for feats, label in zip(features, labels):
            # Calculate the linear combination of weights and features, plus bias
            linear = np.dot(w, feats.transpose()) + b

            # Apply the step activation function
            act = step(linear)

            # Calculate the difference between the actual label and the prediction
            delta = label.item(0, 0) - act

            # If there is a misclassification (delta not 0), update weights and increment misclassified count
            if delta:
                misclassified += 1
                w += delta * feats  # Update weights towards correct classification

        # Append the count of misclassified examples to the results list for this epoch. Indexes represent epochs
        results.append(misclassified)

    # Return the list of misclassification counts per epoch
    return results


print(perceptron(data, 5))
