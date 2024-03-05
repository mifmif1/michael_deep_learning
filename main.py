import numpy as np

from model.smalls.initializations import Initializations
from model.weight.weight_model import Weights

if __name__ == '__main__':
    initial = np.array([1, 2, 3, 4, 5])
    w1 = Weights(previous_length=initial, next_length=4, initialization=Initializations.RANDOM)
    w2 = Weights(previous_length=w1.A, next_length=2, initialization=Initializations.RANDOM)
    print(w2.A)
