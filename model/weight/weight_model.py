from model.smalls.initializations import Initializations
from model.weight.weights_base_model import WeightBaseModel


class Weights(WeightBaseModel):
    def __init__(self,
                 previous_neurons_num: int,
                 next_neurons_num: int,
                 initialization: Initializations = Initializations.ZEROS):
        super().__init__(next_neurons_num=next_neurons_num, previous_neuron_num=previous_neurons_num, initialization=initialization)