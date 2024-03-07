import numpy as np

from model.activations.base_activation import Activation


class Sigmoid(Activation):

    def activate(self, array: np.ndarray):
        return 1 / (1 + np.exp(-array))

    def derivative(self, array: np.ndarray):
        return np.multiply(self.activate(array=array), 1 - self.activate(array=array))
