import numpy as np

from model.costs.base_cost import BaseCost


class MSE(BaseCost):
    def cost(self, y: np.ndarray, y_hat: np.ndarray):
        return np.mean(np.power((y - y_hat), 2))

    def cost_derivative(self, y: np.ndarray, y_hat: np.ndarray):
        pass

