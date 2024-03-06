from typing import List

import numpy as np

from model.layer.layer_model import Layer
from model.smalls.helpers import activate


class Network:
    def __init__(self,
                 data: np.ndarray,
                 layers: List[Layer] = None):
        self._data = data
        self._layers = layers if layers else []

    @property
    def layers(self):
        return self._layers

    def forward_propagation(self) -> np.ndarray:
        if len(self._layers) == 1:
            A = np.add(np.matmul(self._layers[0].weights.values, self._data), self._layers[0].bias.values)
            return activate(A, self._layers[0].activation)
        return activate(
            np.add(np.matmul(Network(data=self._data, layers=self._layers[:-1]).forward_propagation(),
                             self._layers[-1].weights.values), self._layers[-1].bias.values),
            self._layers[-1].activation)

    def back_propagation(self):
        pass
