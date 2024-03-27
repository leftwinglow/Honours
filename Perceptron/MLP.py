import numpy as np


class MLP:
    def __init__(self, layer_sizes: list[int]) -> None:
        self.weights = []
        self.biasees = []
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]))
            self.biases.append(np.random.randn(1, layer_sizes[i + 1]))
        pass
