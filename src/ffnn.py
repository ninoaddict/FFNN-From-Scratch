import pickle
from typing import List, Literal, Union
import matplotlib.pyplot as plt
import numpy as np
from lossfunc import LossFunctions
from layer import Layer
import networkx as nx
from adjustText import adjust_text
from tqdm import tqdm

class FFNN:
    def __init__(
        self,
        layer_sizes: List[int],
        activations: List[Literal["linear", "relu", "sigmoid", "tanh", "softmax", "swish", "elu"]],
        weight_init: Literal["zero", "uniform", "normal", "he", "xavier"]="uniform",
        lower=-0.5,
        upper=0.5,
        mean=0,
        variance=1,
        seed=None,
        use_rmsnorm=False
    ):
        self.layers = []
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.num_classes = layer_sizes[len(layer_sizes) - 1]
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                Layer(
                    layer_sizes[i],
                    layer_sizes[i + 1],
                    activations[i],
                    weight_init,
                    lower,
                    upper,
                    mean,
                    variance,
                    seed,
                    use_rmsnorm
                )
            )

    def __forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def __backward(self, loss_grad):
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad)

    def __update_weights(
        self, learning_rate: float, regularization=None, reg_lambda=0.0
    ):
        for layer in self.layers:
            if regularization == "l1":
                reg_l1 = reg_lambda * np.sign(layer.weights)
                layer.weights -= learning_rate * (layer.grad_weights + reg_l1)
                layer.biases -= learning_rate * layer.grad_biases
                if layer.use_rmsnorm:
                    layer.rmsnorm.g -= learning_rate * layer.rmsnorm.grad_g
            elif regularization == "l2":
                reg_l2 = reg_lambda * 2 * layer.weights
                layer.weights -= learning_rate * (layer.grad_weights + reg_l2)
                layer.biases -= learning_rate * layer.grad_biases
                if layer.use_rmsnorm:
                    layer.rmsnorm.g -= learning_rate * layer.rmsnorm.grad_g
            else:
                layer.weights -= learning_rate * layer.grad_weights
                layer.biases -= learning_rate * layer.grad_biases
                if layer.use_rmsnorm:
                    layer.rmsnorm.g -= learning_rate * layer.rmsnorm.grad_g

    def train(
        self,
        x_train,
        y_train,
        x_val,
        y_val,
        loss_function:Literal["mse", "binary_cross_entropy", "categorical_cross_entropy"]="mse",
        learning_rate=0.01,
        epochs=100,
        batch_size=32,
        verbose=1,
        regularization:Union[None, Literal["l1", "l2"]]=None,
        reg_lambda=0.0,
    ):
        loss_fn = getattr(LossFunctions, loss_function)
        history = {
            "training_loss": [],
            "val_loss": [],
        }
        num_samples = x_train.shape[0]

        if loss_function == "categorical_cross_entropy":
            if len(y_train.shape) == 1:
                y_train = np.eye(self.num_classes)[y_train]
            if len(y_val.shape) == 1:
                y_val = np.eye(self.num_classes)[y_val]
        else:
            if len(y_train.shape) == 1:
                y_train = y_train.reshape(len(y_train), 1)
            if len(y_val.shape) == 1:
                y_val = y_val.reshape(len(y_val), 1)

        for epoch in range(epochs):
            # shuffle the dataset for each epoch
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            x_shuffled, y_shuffled = x_train[indices], y_train[indices]

            # train process
            epoch_loss = 0
            num_batches = (num_samples + batch_size - 1) // batch_size
            if verbose:
                pbar = tqdm(
                    total=num_batches, desc=f"Epoch {epoch+1}/{epochs}", unit="batch"
                )
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples - 1)
                X_batch, y_batch = x_shuffled[start:end], y_shuffled[start:end]

                preds = self.__forward(X_batch)
                loss = loss_fn(y_batch, preds)
                loss_grad = loss_fn(y_batch, preds, derivative=True)

                # add regularization to the loss function
                if regularization == "l1":
                    for layer in self.layers:
                        loss += reg_lambda * (np.sum(np.abs(layer.weights)))
                elif regularization == "l2":
                    for layer in self.layers:
                        loss += reg_lambda * (np.sum(layer.weights**2))

                self.__backward(loss_grad)
                self.__update_weights(learning_rate, regularization, reg_lambda)
                epoch_loss += loss
                if verbose:
                    pbar.update(1)
            if verbose:
                pbar.close()

            epoch_loss /= num_samples // batch_size
            history["training_loss"].append(epoch_loss)

            # compute validation loss
            val_preds = self.__forward(x_val)
            val_loss = loss_fn(y_val, val_preds)
            history["val_loss"].append(val_loss)

            if verbose:
                print(
                    f"Epoch {epoch+1}/{epochs} - Training Loss: {epoch_loss:.4f} - Validation Loss: {val_loss:.4f}"
                )
        return history["training_loss"], history["val_loss"]

    def predict(self, x):
        return self.__forward(x)

    def predict_class(self, x):
        return np.argmax(self.__forward(x), axis=1)

    def save_model(self, filename: str):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(filename: str):
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

    def display_graph(self):
        G = nx.DiGraph()
        pos = {}
        node_cnt = 0
        layer_x_offset = 1

        for layer_idx, num_neurons in enumerate(self.layer_sizes):
            for neuron_idx in range(num_neurons):
                G.add_node(node_cnt, Layer=layer_idx)
                pos[node_cnt] = (layer_x_offset * layer_idx, -neuron_idx)
                node_cnt += 1

        node_cnt = 0
        for layer_idx in range(len(self.layers)):
            num_neurons_curr = self.layer_sizes[layer_idx]
            num_neurons_next = self.layer_sizes[layer_idx + 1]
            for i in range(num_neurons_curr):
                for j in range(num_neurons_next):
                    weight = self.layers[layer_idx].weights[i, j]
                    G.add_edge(
                        node_cnt + i, node_cnt + num_neurons_curr + j, weight=weight
                    )
            node_cnt += num_neurons_curr

        plt.figure(figsize=(8, 6))
        labels = {node: f"N{node}" for node in G.nodes()}
        edge_labels = {(i, j): f'{d["weight"]:.2f}' for i, j, d in G.edges(data=True)}

        nx.draw(
            G,
            pos,
            with_labels=True,
            labels=labels,
            node_size=700,
            node_color="lightblue",
        )
        text_labels = []
        for (i, j), label in edge_labels.items():
            x, y = (pos[i][0] + pos[j][0]) / 2, (pos[i][1] + pos[j][1]) / 2  # Midpoint
            text_labels.append(plt.text(x, y, label, fontsize=8))
        adjust_text(
            text_labels,
        )
        plt.title("Feed Forward Neural Network Visualization")
        plt.show()
