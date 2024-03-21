import numpy as np


class BaseCost:
    def cost(self, y: np.ndarray, y_hat: np.ndarray):
        """
        :param y: shape = (samples, options for y)
        :param y_hat: shape = (samples, options for y)
        :return: Mean Absolute Error
        """
        pass
