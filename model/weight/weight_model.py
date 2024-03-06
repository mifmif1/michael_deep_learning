import numpy as np

from model.smalls.activations import Activations
from model.smalls.initializations import Initializations
from model.weight.weights_base_model import WeightBaseModel


class Weights(WeightBaseModel):
    def __init__(self,
                 previous_length: int,
                 next_length: int,
                 initialization: Initializations = Initializations.ZEROS):
        super().__init__(next_length=next_length, columns=previous_length, initialization=initialization)