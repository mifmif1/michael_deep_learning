from enum import Enum

import numpy as np

from model.activations.base_activation import Activation
from model.activations.relu import Relu
from model.activations.sigmoid import Sigmoid
from model.activations.softmax import Softmax
from model.activations.tanh import Tanh


class Activations(Activation, Enum):
    RELU = Relu()
    TANH = Tanh()
    SIGMOID = Sigmoid()
    SOFTMAX = Softmax()


activ1 = Relu().activate(array=np.array([11, -11]))  # output: in relu
activ2 = Activations.RELU.activate(array=np.array([11, -11]))  # output: in activation
