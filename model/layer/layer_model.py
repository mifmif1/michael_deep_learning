import numpy as np

from model.activations.activations_enum import Activations
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
        self._layer_length = layer_length
        self._activation = activation
        self._weights = Weights(previous_neurons_num=previous_length, next_neurons_num=layer_length,
                                initialization=initialization)
        self._bias = Bias(next_neurons_num=layer_length, initialization=initialization)
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

    @property
    def layer_length(self):
        return self._layer_length