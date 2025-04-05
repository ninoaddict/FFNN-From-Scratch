import numpy as np

class ActivationFunctions:
    @staticmethod
    def linear(x, output=None, derivative=False):
        return x if not derivative else np.ones_like(x)

    @staticmethod
    def relu(x, output=None, derivative=False):
        if derivative:
            return np.where(x > 0, 1, 0)
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x, output=None, derivative=False):
        if derivative:
            return output * (1 - output)
        sig = 1 / (1 + np.exp(-x))
        return sig

    @staticmethod
    def tanh(x, output=None, derivative=False):
        if derivative:
            return 1 - output**2
        t = np.tanh(x)
        return t

    @staticmethod
    def softmax(x, output=None, derivative=False):
        if derivative:
            batch_size, n_classes = output.shape
            jacobians = np.empty((batch_size, n_classes, n_classes))
            for i in range(batch_size):
                s = output[i].reshape(-1, 1)
                jacobians[i] = np.diagflat(s) - np.dot(s, s.T)
            return jacobians
        
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        softmax_x = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return softmax_x
    
    @staticmethod
    def swish(x, output=None, derivative=False):
        if derivative:
            sig = 1 / (1 + np.exp(-x))
            return sig + x * sig * (1 - sig)
        sig = 1 / (1 + np.exp(-x))
        return x * sig

    
    @staticmethod
    def elu(x, output=None, derivative=False, alpha=1.0):
        if derivative:
            turunan = np.where(x >= 0, 1, alpha * np.exp(x))
            return turunan
        return np.where(x >= 0, x, alpha * (np.exp(x) - 1))
