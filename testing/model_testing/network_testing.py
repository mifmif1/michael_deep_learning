import numpy as np

from model.network.layer_helper import LayerNetworkInfo
from model.network.network_model import Network
from model.activations import Activations
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

    Y_hat = my_net.forward_propagation()
    print(np.sum(Y_hat))