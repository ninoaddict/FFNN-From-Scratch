import numpy as np

class ActivationFunctions:
    @staticmethod
    def linear(x, derivative=False):
        return x if not derivative else np.ones_like(x)

    @staticmethod
    def relu(x, derivative=False):
        if derivative:
            return np.where(x > 0, 1, 0)
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x, derivative=False):
        sig = 1 / (1 + np.exp(-x))
        return sig if not derivative else sig * (1 - sig)

    @staticmethod
    def tanh(x, derivative=False):
        t = np.tanh(x)
        return t if not derivative else 1 - t**2

    @staticmethod
    def softmax(x, derivative=False):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        softmax_x = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return softmax_x if not derivative else softmax_x * (1 - softmax_x)