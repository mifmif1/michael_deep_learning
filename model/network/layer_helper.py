from pydantic import BaseModel

from model.activations.activations_enum import Activations
from model.smalls.initializations import Initializations


class LayerNetworkInfo(BaseModel):
    layer_length: int
    activation: Activations = Activations.RELU
    initialization: Initializations = Initializations.ZEROS
