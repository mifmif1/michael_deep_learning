import numpy as np

from model.activations.base_activation import Activation


class Softmax(Activation):

    def activate(self, array: np.ndarray):
        return np.exp(array) / np.sum(np.exp(array))

    def derivative_a(self, array: np.ndarray):
        jacobian_m = np.diag(array)
        for i in range(len(jacobian_m)):
            for j in range(len(jacobian_m)):
                if i == j:
                    jacobian_m[i][j] = array[i] * (1 - array[i])
                else:
                    jacobian_m[i][j] = -array[i] * array[j]
        return jacobian_m

    def derivative(self):
        s = self._array.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)
