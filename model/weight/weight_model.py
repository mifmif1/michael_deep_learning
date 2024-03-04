import numpy as np

from model.smalls.activations import Activations
from model.smalls.initializations import Initializations


class WeightsLayer:
    def __init__(self, previous_layer: np.array | WeightsLayer,
                 length: int = 10,
                 initialization: Initializations = Initializations.ZEROS,
                 activation: Activations = Activations.RELU):
        self._activation = activation
        match initialization:
            case Initializations.ZEROS:
                self._weights = np.zeros(length)
                self._bias = np.zeros(length)
            case Initializations.RANDOM:
                self._weights = np.random.rand(length)
                self._bias = np.random.rand(length)
        self._A = self.activate(previous_layer=previous_layer)

    @property
    def weights(self):
        return self._weights

    @property
    def bias(self):
        return self._bias

    def activate(self, previous_layer):
        A = np.add(np.matmul(previous_layer.weights, self._weights), self._bias)
        match self._activation:
            case Activations.RELU:
                A = np.maximum(A)
            case Activations.TANH:
                A = np.tanh(A)
            case Activations.SIGMOID:
                A = (1 / (1 + np.exp(-A)))
        return A
