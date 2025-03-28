import numpy as np
from actfunc import ActivationFunctions
from rmsnorm import RMSNorm


class Layer:
    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: str,
        weight_init: str,
        lower: float,
        upper: float,
        mean: float,
        variance: float,
        seed,
        use_rmsnorm: bool=True
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
            self.weights = np.random.normal(
                mean, np.sqrt(variance), (input_size, output_size)
            )
            self.biases = np.random.normal(mean, np.sqrt(variance), (1, output_size))
        elif weight_init == "he":
            self.weights = np.random.normal(
                0, np.sqrt(2 / input_size), (input_size, output_size)
            )
            self.biases = np.zeros((1, output_size))
        elif weight_init == "xavier":
            limit = np.sqrt(6 / (input_size + output_size))
            self.weights = np.random.uniform(-limit, limit, (input_size, output_size))
            self.biases = np.zeros((1, output_size))
        else:
            raise ValueError("Unknown weight initialization method")

        self.input = None
        self.output = None
        self.grad_weights = np.zeros((input_size, output_size))
        self.grad_biases = np.zeros((1, output_size))
        
        self.use_rmsnorm = use_rmsnorm
        if self.use_rmsnorm:
            self.rmsnorm = RMSNorm(output_size)
        else:
            self.rmsnorm = None

    def forward(self, x):
        self.input = x
        z = x @ self.weights
        if self.use_rmsnorm:
            z = self.rmsnorm.forward(z)
        z = z + self.biases
        self.output = self.activation(z)
        return self.output

    def backward(self, grad_output):
        if self.activation_name == "softmax":
            grad_list = []
            for i in range(self.output.shape[0]):
                s = self.output[i].reshape(-1, 1)
                jacobian = np.diagflat(s) - np.dot(s, s.T)
                grad_i = np.dot(grad_output[i : i + 1], jacobian)
                grad_list.append(grad_i)
            grad = np.concatenate(grad_list, axis=0)
        else:
            activation_grad = self.activation(self.output, derivative=True)
            grad = grad_output * activation_grad
        
        # calculate the bias grad
        self.grad_biases = np.sum(grad, axis=0, keepdims=True)

        # calculate the weight grad and g grad if applicable
        if self.use_rmsnorm:
            grad = self.rmsnorm.backward(grad)        
        
        self.grad_weights = self.input.T @ grad
        return grad @ self.weights.T
