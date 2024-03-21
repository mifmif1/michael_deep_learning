from dataclasses import dataclass

from model.activations.activations_enum import Activations
from model.smalls.initializations import Initializations


@dataclass
class LayerNetworkInfo:
    layer_length: int
    activation: Activations = Activations.RELU
    initialization: Initializations = Initializations.ZEROS
