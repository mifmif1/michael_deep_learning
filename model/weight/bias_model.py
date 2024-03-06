import numpy as np

from model.smalls.initializations import Initializations
from model.weight.weights_base_model import WeightBaseModel


class Bias(WeightBaseModel):
    def __init__(self,
                 next_length: int = 10,
                 initialization: Initializations = Initializations.ZEROS):
        super().__init__(next_length=next_length, columns=1, initialization=initialization)