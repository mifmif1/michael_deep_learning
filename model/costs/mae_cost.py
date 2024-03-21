import numpy as np

from model.costs.base_cost import BaseCost


class MAE(BaseCost):
    def cost(self, y: np.ndarray, y_hat: np.ndarray):
        """
        :param y: shape = (samples, options for y)
        :param y_hat: shape = (samples, options for y)
        :return: Mean Absolute Error
        """
        return np.mean(np.absolute(y - y_hat))
