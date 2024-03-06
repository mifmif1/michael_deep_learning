import numpy as np

from model.smalls.initializations import Initializations


class WeightBaseModel:
    def __init__(self,
                 next_neurons_num: int,
                 previous_neuron_num: int,
                 initialization: Initializations = Initializations.ZEROS):
        self._next_neurons_num = next_neurons_num
        self._previous_neurons_num = previous_neuron_num
        self._initialization = initialization
        self._values = self.initialize()

    def initialize(self) -> np.array:
        match self._initialization:
            case Initializations.ZEROS:
                return np.zeros((self._previous_neurons_num, self._next_neurons_num))
            case Initializations.RANDOM:
                return np.random.rand(self._previous_neurons_num, self._next_neurons_num)

    @property
    def values(self) -> np.array:
        return self._values
