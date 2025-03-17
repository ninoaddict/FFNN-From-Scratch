import pickle
import matplotlib.pyplot as plt
import numpy as np
from lossfunc import LossFunctions
from layer import Layer


class FFNN:
    def __init__(
        self,
        layer_sizes,
        activations,
        weight_init="uniform",
        lower=-0.1,
        upper=0.1,
        seed=None,
    ):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                Layer(
                    layer_sizes[i],
                    layer_sizes[i + 1],
                    activations[i],
                    weight_init,
                    lower,
                    upper,
                    seed,
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss_grad):
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad)

    def update_weights(self, learning_rate):
        for layer in self.layers:
            layer.weights -= learning_rate * layer.grad_weights
            layer.biases -= learning_rate * layer.grad_biases
    
    def rms_norm(self):
        for layer in self.layers:
            mean = np.mean(layer.weights)
            variance = np.var(layer.weights)
            layer.weights = (layer.weights - mean) / np.sqrt(variance + 1e-6)

    def train(
        self,
        X,
        y,
        loss_function="mse",
        learning_rate=0.01,
        epochs=100,
        batch_size=32,
        verbose=1,
        rms_norm=False
    ):
        loss_fn = getattr(LossFunctions, loss_function)
        history = []
        num_samples = X.shape[0]

        for epoch in range(epochs):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_shuffled, y_shuffled = X[indices], y[indices]

            epoch_loss = 0
            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                X_batch, y_batch = X_shuffled[start:end], y_shuffled[start:end]

                preds = self.forward(X_batch)
                loss = loss_fn(y_batch, preds)
                loss_grad = loss_fn(y_batch, preds, derivative=True)
                self.backward(loss_grad)
                self.update_weights(learning_rate)
                if rms_norm:
                    self.rms_norm()
                epoch_loss += loss

            epoch_loss /= num_samples // batch_size
            history.append(epoch_loss)

            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
        return history

    def save_model(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

    def plot_weight_distribution(self, layers):
        for i in layers:
            weights = self.layers[i].weights.flatten()
            plt.hist(weights, bins=30, alpha=0.7, label=f"Layer {i}")
        plt.legend()
        plt.title("Weight Distribution")
        plt.xlabel("Weight Values")
        plt.ylabel("Frequency")
        plt.show()

    def plot_gradient_distribution(self, layers):
        for i in layers:
            grads = self.layers[i].grad_weights.flatten()
            plt.hist(grads, bins=30, alpha=0.7, label=f"Layer {i}")
        plt.legend()
        plt.title("Gradient Distribution")
        plt.xlabel("Gradient Values")
        plt.ylabel("Frequency")
        plt.show()
