import numpy as np

from model.smalls.initializations import Initializations
from model.weight.weights_base_model import WeightBaseModel


class Bias(WeightBaseModel):
    def __init__(self,
                 next_neurons_num: int = 10,
                 initialization: Initializations = Initializations.ZEROS):
        super().__init__(next_neurons_num=next_neurons_num, previous_neuron_num=1, initialization=initialization)
