from typing import List

import numpy as np

from model.costs.costs_enum import Costs
from model.layer.layer_model import Layer
from model.network.layer_helper import LayerNetworkInfo


class Network:
    def __init__(self,
                 data: np.ndarray,
                 cost_function: Costs,
                 layers: List[Layer] = None):
        self._data = data
        self._cost_function = cost_function
        self._layers = layers if layers else []

    @property
    def layers(self):
        return self._layers

    def add_layer(self, layer_network_info: LayerNetworkInfo):
        previous_length = self._layers[-1].layer_length if self._layers else self._data.size
        layer = Layer(previous_layer_neurons_num=previous_length,
                      neurons_num=layer_network_info.layer_length,
                      activation=layer_network_info.activation,
                      initialization=layer_network_info.initialization)
        self._layers.append(layer)

    def forward_propagation(self) -> np.ndarray:
        assert len(self._layers) >= 1, "You must define layers before forward propagation"
        if len(self._layers) == 1:
            self._layers[0].Z = np.add(np.matmul(self._data, self._layers[0].weights.values),
                                       self._layers[0].bias.values)
            self._layers[0].A = self._layers[0].activation.activate(self._layers[0].Z)
            return self._layers[0].A
        self._layers[-1].Z = np.add(np.matmul(Network(data=self._data, layers=self._layers[:-1]).forward_propagation(),
                                              self._layers[-1].weights.values), self._layers[-1].bias.values)
        self._layers[-1].A = self._layers[-1].activation.activate(self._layers[-1].Z)
        return self._layers[-1].A

    def _cost(self, y_hat: np.ndarray):
        return self._cost_function.value.cost(y=self._layers[-1].A, y_hat=y_hat)

    def _cost_derivative(self, y_hat: np.ndarray):
        return self._cost_function.value.cost_derivative()

    def back_propagation(self, y_hat: np.ndarray):
        cost = self._cost(y_hat=y_hat)



        # todo : complete, by inspiration of by https://github.com/pranavbudhwant/backpropagation-in-numpy/blob/master/implementation.py
        pass
