import numpy as np

from model.smalls.activations import Activations
from model.smalls.initializations import Initializations
from model.weight.weight_model import Weights
from model.network.network_model import Network
from model.layer.layer_model import Layer

if __name__ == '__main__':
    initial = np.array([1, 2, 3, 4, 5])
    my_net = Network(data=initial)
    my_net.layers.append(Layer(previous_length=initial.size, layer_length=5, activation=Activations.TANH,
                               initialization=Initializations.RANDOM))
    my_net.layers.append(Layer(previous_length=initial.size, layer_length=5, activation=Activations.RELU,
                               initialization=Initializations.RANDOM))
    my_net.layers.append(Layer(previous_length=initial.size, layer_length=5, activation=Activations.SIGMOID,
                               initialization=Initializations.RANDOM))
    print(my_net.forward_propagation())
