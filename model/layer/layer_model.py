import numpy as np

from model.activations.activations_enum import Activations
from model.smalls.initializations import Initializations
from model.weight.bias_model import Bias
from model.weight.weight_model import Weights


class Layer:
    def __init__(self,
                 previous_layer_neurons_num: int,
                 neurons_num: int,
                 activation: Activations = Activations.RELU,
                 initialization: Initializations = Initializations.ZEROS):
        self._previous_layer_neurons_num = previous_layer_neurons_num
        self._neurons_num = neurons_num
        self._activation = activation
        self._weights = Weights(previous_neurons_num=previous_layer_neurons_num, next_neurons_num=neurons_num,
                                initialization=initialization)
        self._bias = Bias(next_neurons_num=neurons_num, initialization=initialization)
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
        return self._neurons_num
