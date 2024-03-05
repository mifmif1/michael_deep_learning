import numpy as np

from model.smalls.activations import Activations


def activate(array: np.ndarray, activation: Activations) -> np.ndarray:
    match activation:
        case Activations.RELU:
            array = np.maximum(0, array)
        case Activations.TANH:
            array = np.tanh(array)
        case Activations.SIGMOID:
            array = (1 / (1 + np.exp(-array)))
    return array
