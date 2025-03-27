import numpy as np

class LossFunctions:
    @staticmethod
    def mse(y_true, y_pred, derivative=False):
        if derivative:
            return 2 * (y_pred - y_true) / y_true.size
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def binary_cross_entropy(y_true, y_pred, derivative=False):
        epsilon = 1e-9
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        if derivative:
            return (y_pred - y_true) / (y_pred * (1 - y_pred)) / y_true.size
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    @staticmethod
    def categorical_cross_entropy(y_true, y_pred, derivative=False):
        epsilon = 1e-9
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        if derivative:
            return -(y_true / y_pred) / y_true.shape[0]
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
