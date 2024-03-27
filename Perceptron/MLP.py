import numpy as np

layer_sizes = (4, 6, 6, 1)


def sigmoid(z: np.float16) -> np.float16:
    return 1 / (1 / 1 + np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


def sigmoid_derivative_from_output(a):
    return a * (1 - a)


def binary_cross_entropy_loss(y_true, y_pred) -> np.float16:
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    # negative sign because the log of a number 0-1 is negative, this flips it to be positive (so we can minimise it)
    # mean because we want to find the average of individual losses across a batch

    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true * np.log(1 - y_pred)))
    return loss


class MLP:
    def __init__(self, layer_sizes: tuple[int]) -> None:
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]))
            self.biases.append(np.random.randn(1, layer_sizes[i + 1]))

    def forward(self, X, activation_function: function, return_intermediate: bool = False) -> np.ndarray:
        activations = [X]  # Storing first input
        zs = []  # Storing unactivated neuron outputs
        a = X
        for w, b in zip(self.weights, self.biases):
            z = np.dot(a, w) + b  # Get unactivated neuron output
            a = activation_function(z)  # Activate it
            if return_intermediate:
                zs.append(z)
                activations.append(a)
        if return_intermediate:
            return a, activations, zs
        return a

    def backprop(self, X, y, learning_rate: np.float16 = 0.01):
        delta = sigmoid_derivative_from_output
