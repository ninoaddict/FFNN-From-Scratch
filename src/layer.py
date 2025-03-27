import numpy as np
from actfunc import ActivationFunctions

class Layer:
    def __init__(
        self, input_size, output_size, activation, weight_init, lower, upper, mean, variance, seed
    ):
        self.activation_name = activation
        self.activation = getattr(ActivationFunctions, activation)

        if seed is not None:
            np.random.seed(seed)

        if weight_init == "zero":
            self.weights = np.zeros((input_size, output_size))
            self.biases = np.zeros((1, output_size))
        elif weight_init == "uniform":
            self.weights = np.random.uniform(lower, upper, (input_size, output_size))
            self.biases = np.random.uniform(lower, upper, (1, output_size))
        elif weight_init == "normal":
            self.weights = np.random.normal(mean, np.sqrt(variance), (input_size, output_size))
            self.biases = np.random.normal(mean, np.sqrt(variance), (1, output_size))
        else:
            raise ValueError("Unknown weight initialization method") # add more for bonus

        self.input = None
        self.output = None
        self.grad_weights = None
        self.grad_biases = None

    def forward(self, x):
        self.input = x
        self.output = self.activation(x @ self.weights + self.biases)
        return self.output

    def backward(self, grad_output):
        if self.activation_name == 'softmax':
            grad_list = []
            for i in range(self.output.shape[0]):
                s = self.output[i].reshape(-1, 1)
                jacobian = np.diagflat(s) - np.dot(s, s.T)
                grad_i = np.dot(grad_output[i:i+1], jacobian)
                grad_list.append(grad_i)
            grad = np.concatenate(grad_list, axis=0)
        else:
            activation_grad = self.activation(self.output, derivative=True)
            grad = grad_output * activation_grad

        self.grad_weights = self.input.T @ grad
        self.grad_biases = np.sum(grad, axis=0, keepdims=True)

        return grad @ self.weights.T