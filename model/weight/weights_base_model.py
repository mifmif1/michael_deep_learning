import numpy as np

from model.smalls.initializations import Initializations


class WeightBaseModel:
    def __init__(self,
                 next_length: int,
                 columns: int,
                 initialization: Initializations = Initializations.ZEROS):
        self._next_length = next_length
        self._columns = columns
        self._initialization = initialization
        self._values = self.initialize()

    def initialize(self) -> np.array:
        match self._initialization:
            case Initializations.ZEROS:
                return np.zeros((self._columns, self._next_length))
            case Initializations.RANDOM:
                return np.random.rand(self._columns, self._next_length)

    @property
    def values(self) -> np.array:
        return self._values
