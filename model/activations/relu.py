import numpy as np

from model.activations.base_activation import Activation


class Relu(Activation):
    def activate(self, array: np.ndarray):
        print('in relu')
        return np.maximum(0, array)

    def derivative(self, array: np.ndarray):
        return np.where(array <= 0, 0, 1)
