import numpy as np

from model.activations.activations_enum import Activations
from model.network.layer_helper import LayerNetworkInfo
from model.network.network_model import Network
from model.smalls.initializations import Initializations


def net_test():
    initial = np.array([1, 2, 3, 4, 5, 6])
    initial = initial / np.sum(initial)
    my_net = Network(data=initial)
    my_net.add_layer(
        LayerNetworkInfo(layer_length=5, activation=Activations.TANH, initialization=Initializations.RANDOM))
    my_net.add_layer(
        LayerNetworkInfo(layer_length=8, activation=Activations.RELU, initialization=Initializations.RANDOM))
    my_net.add_layer(
        LayerNetworkInfo(layer_length=40, activation=Activations.RELU, initialization=Initializations.RANDOM))
    my_net.add_layer(
        LayerNetworkInfo(layer_length=40, activation=Activations.SOFTMAX, initialization=Initializations.RANDOM))
    my_net.add_layer(
        LayerNetworkInfo(layer_length=8, activation=Activations.TANH, initialization=Initializations.RANDOM))
    my_net.add_layer(
        LayerNetworkInfo(layer_length=2, activation=Activations.SOFTMAX, initialization=Initializations.RANDOM))

    net_outcome = my_net.forward_propagation()
    print(f"{net_outcome=}\n{np.argmax(net_outcome)=}")
