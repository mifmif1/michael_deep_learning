from enum import Enum

import numpy as np

from model.activations.base_activation import Activation
from model.activations.relu import Relu
from model.activations.sigmoid import Sigmoid
from model.activations.softmax import Softmax
from model.activations.tanh import Tanh


class Activations(Activation, Enum):
    RELU: Activation = Relu
    TANH: Activation = Tanh
    SIGMOID: Activation = Sigmoid
    SOFTMAX: Activation = Softmax


def foo(activ: Activations, array: np.ndarray):
    return activ.activate(array)


a = Activations.RELU
array = np.array([1, -1, 2])
b = a.activate(array=np.array([1, -1, 2]))

foo(a, np.array([1, -1, 2]))
print(b)
