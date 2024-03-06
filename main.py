import numpy as np

from model.layer.layer_model import Layer
from model.network.network_model import Network
from model.smalls.activations import Activations
from model.smalls.initializations import Initializations

if __name__ == '__main__':
    initial = np.array([1, 2, 3, 4, 5])
    my_net = Network(data=initial)
    l1 = Layer(previous_length=initial.size, layer_length=6, activation=Activations.TANH,
               initialization=Initializations.RANDOM)
    l2 = Layer(previous_length=l1.layer_length, layer_length=7, activation=Activations.RELU,
               initialization=Initializations.RANDOM)
    l3 = Layer(previous_length=l2.layer_length, layer_length=2, activation=Activations.SIGMOID,
               initialization=Initializations.RANDOM)
    print(my_net.forward_propagation())
