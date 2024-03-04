import numpy as np

from model.smalls.activations import Activations
from model.smalls.initializations import Initializations


class WeightsLayer:
    def __init__(self, previous_layer_A: np.array,
                 next_length: int = 10,
                 initialization: Initializations = Initializations.ZEROS,
                 activation: Activations = Activations.RELU):
        self._activation = activation
        match initialization:
            case Initializations.ZEROS:
                self._weights = np.zeros((next_length, previous_layer_A.size))
                self._bias = np.zeros(next_length)
            case Initializations.RANDOM:
                self._weights = np.random.rand(previous_layer_A.size)
                self._bias = np.random.rand(next_length)
        self._A = self.activate(previous_layer_A=previous_layer_A)

    @property
    def weights(self):
        return self._weights

    @property
    def bias(self):
        return self._bias

    @property
    def A(self):
        return self._A

    def activate(self, previous_layer_A: np.array):
        A = np.add(np.matmul(self._weights, previous_layer_A), self._bias)
        match self._activation:
            case Activations.RELU:
                self._A = np.maximum(0, A)
            case Activations.TANH:
                self._A = np.tanh(A)
            case Activations.SIGMOID:
                self._A = (1 / (1 + np.exp(-A)))
        return self._A
