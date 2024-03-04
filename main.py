import numpy as np

from model.smalls.initializations import Initializations
from model.weight.weight_model import WeightsLayer

if __name__ == '__main__':
    initial = np.array([1, 2, 3, 4, 5])
    w1 = WeightsLayer(previous_layer_A=initial, next_length=4, initialization=Initializations.RANDOM)
    w2 = WeightsLayer(previous_layer_A=w1.A, next_length=2, initialization=Initializations.RANDOM)
    print(w2.A)
