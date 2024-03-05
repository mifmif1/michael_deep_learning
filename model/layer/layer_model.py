import numpy as np

from model.smalls.activations import Activations
from model.smalls.initializations import Initializations
from model.weight.bias_model import Bias
from model.weight.weight_model import Weights


class Layer:
    def __init__(self,
                 previous_length: int,
                 layer_length: int,
                 activation: Activations = Activations.RELU,
                 initialization: Initializations = Initializations.ZEROS):
        self._previous_length = previous_length
        self._activation = activation
        self._initialization = initialization
        self._weights = Weights(previous_length=previous_length, next_length=layer_length,
                                initialization=initialization)
        self._bias = Bias(next_length=layer_length, initialization=initialization)
        self._A: np.array

    @property
    def A(self):
        return self._A

    @property
    def weights(self):
        return self._weights

    @property
    def bias(self):
        return self._bias

    @property
    def activation(self):
        return self._activation

    def activate(self, ):
        A = np.add(np.matmul(self._weights.values, previous_layer_A), self._bias.values)
        match self._activation:
            case Activations.RELU:
                self._A = np.maximum(0, A)
            case Activations.TANH:
                self._A = np.tanh(A)
            case Activations.SIGMOID:
                self._A = (1 / (1 + np.exp(-A)))
        return self._A
