from typing import List

import numpy as np

from model.layer.layer_model import Layer
from model.smalls.helpers import activate


class Network:
    def __init__(self,
                 data: np.ndarray,
                 layers: List[Layer]):
        self._data = data
        self._layers = layers

    def forward_propagation(self):
        if len(self._layers) == 1:
            return activate(np.add(np.matmul(self._layers[0].weights.values, self._data), self._layers[0].bias.values),
                            self._layers[0].activation)
        return Network(data=self._data, layers=self._layers[:-1]).forward_propagation()

    def back_propagation(self):
        pass
