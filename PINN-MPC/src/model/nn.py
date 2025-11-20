import abc
import logging
import os
import time
import datetime

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from optimizer.lbfgs import LBFGS
from utils.plotting import new_fig, save_fig

CHECKPOINTS_PATH = os.path.join('../checkpoints')


class Normalize(nn.Module):
    """
    Normalization layer:
        X -> 2 * (X - lb) / (ub - lb) - 1
    """

    def __init__(self, lb: np.ndarray, ub: np.ndarray, dtype=torch.float64, device=None):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        lb_t = torch.as_tensor(lb, dtype=dtype, device=device)
        ub_t = torch.as_tensor(ub, dtype=dtype, device=device)
        self.register_buffer("lb", lb_t)
        self.register_buffer("ub", ub_t)

    def forward(self, X):
        return 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0


class NN(object, metaclass=abc.ABCMeta):
    """
    Abstract class used to represent a Neural Network.
    """

    def __init__(self, layers: list, lb: np.ndarray, ub: np.ndarray) -> None:
        """
        Constructor.

        :param list layers: widths of the layers
        :param np.ndarray lb: lower bounds of the inputs of the training data
        :param np.ndarray ub: upper bounds of the inputs of the training data
        """

        self.checkpoints_dir = CHECKPOINTS_PATH

        # use float64 as in original TF code
        self.dtype = torch.float64
        torch.set_default_dtype(self.dtype)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_dim = layers[0]
        self.output_dim = layers[-1]

        # Build Sequential model: [Normalize] + [Linear + tanh]* + [Linear(out)]
        modules = []

        # Normalization Layer (instead of Keras Lambda)
        modules.append(Normalize(lb, ub, dtype=self.dtype, device=self.device))

        prev_width = self.input_dim
        # Hidden Layers
        for layer_width in layers[1:-1]:
            linear = nn.Linear(prev_width, layer_width)
            # Xavier (Glorot) normal init: same 의도 as 'glorot_normal'
            nn.init.xavier_normal_(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)

            modules.append(linear)
            modules.append(nn.Tanh())
            prev_width = layer_width

        # Output Layer
        out_linear = nn.Linear(prev_width, self.output_dim)
        nn.init.xavier_normal_(out_linear.weight)
        if out_linear.bias is not None:
            nn.init.zeros_(out_linear.bias)
        modules.append(out_linear)

        self.model = nn.Sequential(*modules).to(self.device)

        self.optimizer = None
        self.loss_object = nn.MSELoss()

        self.start_time = None
        self.prev_time = None

        # Store metrics
        self.train_loss_results = {}
        self.train_accuracy_results = {}
        self.train_time_results = {}
        self.train_pred_results = {}

    def tensor(self, X, requires_grad: bool = False):
        """
        Converts a list or numpy array to a torch.Tensor,
        and optionally sets requires_grad.

        :param X: list / np.ndarray / torch.Tensor
        :param requires_grad: whether autograd should track this tensor
        :return: torch.Tensor
        """
        if isinstance(X, torch.Tensor):
            t = X.to(self.device, dtype=self.dtype)
        else:
            t = torch.as_tensor(X, dtype=self.dtype, device=self.device)

        t.requires_grad_(requires_grad)
        return t

    def summary(self):
        """
        Rough equivalent of Keras model.summary() using logging.
        """
        model_str = str(self.model)
        for line in model_str.split("\n"):
            logging.info(line)

    def train_step(self, x, y):
        """
        Performs training step during training.

        :param torch.Tensor x: (batched) input tensor of training data
        :param torch.Tensor y: (batched) output tensor of training data
        :return: float loss: the corresponding current loss value
        """
        self.model.train()
        self.optimizer.zero_grad()

        y_pred = self.model(x)
        loss = self.loss_object(y_pred, y)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def fit(self, x, y, epochs=2000, x_test=None, y_test=None, optimizer='adam', learning_rate=0.1,
            load_best_weights=False, val_freq=1000, log_freq=1000, verbose=1):
        """
        Performs the neural network training phase.

        :param np.ndarray or torch.Tensor x: input tensor of the training dataset
        :param np.ndarray or torch.Tensor y: output tensor of the training dataset
        :param int epochs: number of training epochs
        :param np.ndarray or torch.Tensor x_test: input tensor of the test dataset, used to evaluate current accuracy
        :param np.ndarray or torch.Tensor y_test: output tensor of the test dataset, used to evaluate current accuracy
        :param str optimizer: name of the optimizer, choose from 'adam' or 'lbfgs'
        :param bool load_best_weights: flag to determine if the best weights corresponding to the best
        accuracy are loaded after training
        """

        x = self.tensor(x)
        y = self.tensor(y)

        self.start_time = time.time()
        self.prev_time = self.start_time

        if optimizer == 'adam':
            self.train_adam(x, y, epochs, x_test, y_test, learning_rate, val_freq, log_freq, verbose)
        elif optimizer == 'lbfgs':
            self.train_lbfgs(x, y, epochs, x_test, y_test, learning_rate, val_freq, log_freq, verbose)

        if load_best_weights is True:
            self.load_weights()

    def train_adam(self, x, y, epochs=2000, x_test=None, y_test=None, learning_rate=0.1,
                   val_freq=1000, log_freq=1000, verbose=1):
        """
        Performs the neural network training, using the adam optimizer.

        :param torch.Tensor x: input tensor of the training dataset
        :param torch.Tensor y: output tensor of the training dataset
        :param int epochs: number of training epochs
        :param np.ndarray or torch.Tensor x_test: input tensor of the test dataset, used to evaluate accuracy
        :param np.ndarray or torch.Tensor y_test: output tensor of the test dataset, used to evaluate accuracy
        """

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        if verbose:
            logging.info(f'Start ADAM optimization')

        epoch_loss = 0.0
        for epoch in range(1, epochs + 1):
            epoch_loss = self.train_step(x, y)

            # epoch_loss is last batch loss (we train in single batch anyway)
            self.epoch_callback(epoch, epoch_loss, epochs,
                                x_val=x_test, y_val=y_test,
                                val_freq=val_freq, log_freq=log_freq,
                                verbose=verbose)

    def train_lbfgs(self, x, y, epochs=2000, x_test=None, y_test=None,
                    learning_rate=1.0, val_freq=1000, log_freq=1000, verbose=1):
        """
        Performs the neural network training, using the L-BFGS optimizer.

        :param torch.Tensor x: input tensor of the training dataset
        :param torch.Tensor y: output tensor of the training dataset
        :param int epochs: number of training epochs
        :param np.ndarray or torch.Tensor x_test: input tensor of the test dataset, used to evaluate accuracy
        :param np.ndarray or torch.Tensor y_test: output tensor of the test dataset, used to evaluate accuracy
        """

        # train the model with L-BFGS solver
        if verbose:
            logging.info(f'Start L-BFGS optimization')

        optimizer = LBFGS()
        optimizer.minimize(
            self.model,
            self.loss_object,
            x,
            y,
            self.epoch_callback,
            epochs=epochs,
            learning_rate=learning_rate,
            x_test=x_test,
            y_test=y_test,
            val_freq=val_freq,
            log_freq=log_freq,
            verbose=verbose,
        )

    def predict(self, x):
        """
        Calls the model prediction function and returns the prediction on an input tensor.

        :param np.ndarray or torch.Tensor x: input tensor
        :return: np.ndarray: output array
        """
        self.model.eval()
        with torch.no_grad():
            x_t = self.tensor(x)
            y_t = self.model(x_t)
        return y_t.detach().cpu().numpy()

    def plot_train_results(self, basename=None):
        """
        Visualizes the training metrics Loss resp. Accuracy over epochs.

        :param str basename: used to save the figure with this name, if None the figure is not saved
        """

        fig = new_fig()
        ax = fig.add_subplot(111)
        fig.suptitle(f'{getattr(self, "name", "NN")} - Training Metrics')

        ax.set_ylabel('Loss')
        ax.set_yscale('log')
        ax.plot(self.train_loss_results.keys(), self.train_loss_results.values(), label='Loss')
        if self.train_accuracy_results:
            ax.set_ylabel("Loss / Accuracy")
            ax.plot(self.train_accuracy_results.keys(), self.train_accuracy_results.values(), label='Accuracy')
        ax.set_xlabel("Epoch", fontsize=14)
        ax.legend(loc='best')
        if basename is not None:
            save_fig(fig, f'{basename}_train_metrics')
        fig.tight_layout()
        plt.show()

    def train_results(self):
        """
        Returns the training metrics stored in dictionaries.

        :return: dict: loss over epochs, dict: accuracy over epochs,
        dict: predictions (on the testing dataset) over epochs
        """

        return self.train_loss_results, self.train_accuracy_results, self.train_pred_results

    def reset_train_results(self):
        """
        Clears the training metrics.
        """
        self.train_loss_results = {}
        self.train_accuracy_results = {}
        self.train_pred_results = {}

    def get_weights(self):
        """
        Returns the model weights.

        :return: state_dict of the model
        """
        return self.model.state_dict()

    def set_weights(self, weights):
        """
        Set the model weights.
        :param weights: state_dict compatible with this model
        """
        self.model.load_state_dict(weights)

    def save_weights(self, path):
        """
        Saves the model weights under a specified path.

        :param str path: path where the weights are saved (base name, '.pt' appended if missing)
        """
        base_dir = os.path.dirname(path)
        if base_dir != "":
            Path(base_dir).mkdir(parents=True, exist_ok=True)

        if not path.endswith(".pt"):
            path = path + ".pt"
        torch.save(self.model.state_dict(), path)

    def load_weights(self, path=None):
        """
        Loads the model weights from a specified path.

        :param str path: path where the weights are saved (base name, '.pt' appended if missing),
                         if None, load from default checkpoints_dir/easy_checkpoint.pt
        """

        if path is None:
            path = os.path.join(self.checkpoints_dir, 'easy_checkpoint.pt')
        elif not path.endswith(".pt"):
            path = path + ".pt"

        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        logging.info(f'\tWeights loaded from {path}')

    def get_epoch_duration(self):
        """
        Measures the time for a training epoch.

        :return: float time per epoch (MM:SS.sss)
        """

        now = time.time()
        epoch_duration = datetime.datetime.fromtimestamp(now - self.prev_time).strftime("%M:%S.%f")[:-4]
        self.prev_time = now
        return epoch_duration

    def get_elapsed_time(self):
        """
        Measures the time since training start.

        :return: datetime.timedelta elapsed time
        """

        return datetime.timedelta(seconds=int(time.time() - self.start_time))

    def epoch_callback(self, epoch, epoch_loss, epochs, x_val=None, y_val=None,
                       val_freq=1000, log_freq=1000, verbose=1):
        """
        Callback function, which is called after each epoch, to produce proper training logging
        and keep track of training metrics.

        :param int epoch: current epoch
        :param float epoch_loss: current loss value
        :param int epochs: number of training epochs
        :param np.ndarray or torch.Tensor x_val: input tensor of the test dataset, used to evaluate current accuracy
        :param np.ndarray or torch.Tensor y_val: output tensor of the test dataset, used to evaluate current accuracy
        :param int val_freq: number of epochs passed before trigger validation
        :param int log_freq: number of epochs passed before each logging
        """

        self.train_loss_results[epoch] = float(epoch_loss)
        elapsed_time = self.get_elapsed_time()
        self.train_time_results[epoch] = elapsed_time

        if epoch % val_freq == 0 or epoch == 1:
            length = len(str(epochs))
            log_str = f'\tEpoch: {str(epoch).zfill(length)}/{epochs},\t' \
                      f'Loss: {epoch_loss:.4e}'

            if x_val is not None and y_val is not None:
                mean_squared_error, errors, Y_pred = self.evaluate(x_val, y_val)
                self.train_accuracy_results[epoch] = float(mean_squared_error)
                self.train_pred_results[epoch] = Y_pred
                log_str += f',\tAccuracy (MSE): {mean_squared_error:.4e}'
                if mean_squared_error <= min(self.train_accuracy_results.values()):
                    self.save_weights(os.path.join(self.checkpoints_dir, 'easy_checkpoint'))

            if (epoch % log_freq == 0 or epoch == 1) and verbose == 1:
                log_str += f',\t Elapsed time: {elapsed_time} (+{self.get_epoch_duration()})'
                logging.info(log_str)

        if epoch == epochs and x_val is None and y_val is None:
            self.save_weights(os.path.join(self.checkpoints_dir, 'easy_checkpoint'))

    def evaluate(self, x_val, y_val, metric='MSE'):
        """
        Evaluates accuracy on validation set.
        Returns CPU numpy-safe values so LBFGS callback does not break.
        """

        x_val = self.tensor(x_val)
        y_val = self.tensor(y_val)

        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(x_val)

        if metric == 'MSE':
            errors = (y_val - y_pred) ** 2
        elif metric == 'MAE':
            errors = torch.abs(y_val - y_pred)
        else:
            raise ValueError("Unsupported metric. Choose 'MSE' or 'MAE'.")

        # *** 핵심 수정 ***
        mean_error = errors.mean().item()                    # python float
        errors_cpu = errors.detach().cpu().numpy()           # numpy array
        y_pred_cpu = y_pred.detach().cpu().numpy()           # numpy array

        return mean_error, errors_cpu, y_pred_cpu
    def prediction_time(self, batch_size, executions=1000):
        """
        Helper function to measure the mean prediction time of the neural network.

        :param int batch_size: dummy batch size of the input tensor
        :param int executions: number of performed executions to determine the mean value
        :return: float mean_prediction_time: the mean prediction time of the neural network on all executions
        """
        X = torch.rand(executions, batch_size, self.input_dim, dtype=self.dtype, device=self.device)

        start_time = time.time()
        for execution in range(executions):
            _ = self.predict(X[execution])
        prediction_time = time.time() - start_time
        mean_prediction_time = prediction_time / executions

        return mean_prediction_time
