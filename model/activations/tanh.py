import numpy as np

from model.activations.base_activation import Activation


class Tanh(Activation):
    def activate(self, array: np.ndarray):
        return np.tanh(array)

    def derivative(self, array: np.ndarray):
        return 1 - np.power(np.tanh(array), 2)
