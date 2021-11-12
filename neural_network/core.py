"""
The core contains the definition of the perceptron class and the layer class
"""
# a b c d e f g h i j k l m n o p q r s t u v w x y z
from __future__ import annotations

from typing import Callable
from typing import List
from typing import Optional
from typing import Union

import numpy as np

from neural_network.activators import ActivationFunction
from neural_network.services import to_vector, make_batches, make_normalization_function, mse, pairwise, shuffle_
from neural_network.settings import MINIMUM_LAYER_COUNT


class Dense:
    """
    Fully-connected layer of neurons
    """
    AF_ALLOWED_TYPES = [type(None), str, ActivationFunction]

    def __init__(self,
                 neurons: int,
                 activator: Union[str, ActivationFunction, None] = None,
                 use_bias: bool = False):
        if not isinstance(neurons, int):
            raise TypeError(f"The number of neurons must be a positive integer, not {type(neurons)}")
        if neurons < 0:
            raise ValueError("The number of neurons must be a positive integer, not negative")

        self.__neurons = neurons
        self.__use_bias = use_bias

        self.weights = None

        self.input = None
        self.__output = None
        self.__error = None

        activator_type = type(activator)

        if not activator:
            self.__activation_function = None
            return
        if activator_type == ActivationFunction:
            self.__activation_function = activator
            return
        if activator_type == str:
            self.__activation_function = ActivationFunction.get_by_name(activator)
            return

        raise TypeError(f"Activator must be one of these types: {self.AF_ALLOWED_TYPES}, not {activator_type}")

    @property
    def neurons(self) -> int:
        """
        The number of neurons on this layer (without bias anyway)
        """
        return self.__neurons

    @property
    def use_bias(self) -> bool:
        """
        Whether the bias is used
        """
        return self.__use_bias

    @property
    def activation_function(self) -> Optional[ActivationFunction]:
        """
        Layer activation function
        """
        return self.__activation_function

    @property
    def output(self) -> np.ndarray:
        """
        Input after activation function
        """
        return self.__output

    @output.setter
    def output(self, value):
        # Bias is added for use on the next layer
        if self.use_bias:
            self.__output = np.vstack((value, 1))
        else:
            self.__output = value

    @property
    def error(self) -> np.ndarray:
        """
        Just error used for backpropagation algorithm
        """
        # The error for bias is removed to correctly update the weights on the previous layer
        if self.use_bias:
            return self.__error[:-1, :]
        return self.__error

    @error.setter
    def error(self, value):
        self.__error = value

    def __repr__(self):
        activation_function_name = self.activation_function.name if not self.is_first() else None
        return f"Fully-connected layer [" \
               f"neurons = {self.neurons}, " \
               f"use_bias = {self.use_bias}, " \
               f"First layer? = {self.is_first()}, " \
               f"Last layer? = {self.is_last()}, " \
               f"activation function = {activation_function_name}" \
               f"]"

    def initialize_weights(self, next_layer_neurons: int) -> None:
        """
        Initialising a matrix of weights with a normal distribution

        :param next_layer_neurons: Number of neurons on the next layer
        """
        matrix_size = (next_layer_neurons, self.neurons + 1) if self.use_bias else (next_layer_neurons, self.neurons)
        self.weights = np.random.normal(loc=0.0, scale=pow(matrix_size[1], -0.5), size=matrix_size)

    def is_first(self) -> bool:
        """
         Checks if this is the first layer.

        The first layer has no activation function (and input)
        """
        return self.activation_function is None

    def is_last(self) -> bool:
        """
        Checks if this is the last layer.

        The last layer has no weights
        """
        return self.weights is None


class Perceptron:
    """
    Multilayer perceptron class
    """
    layers: List[Dense]
    learning_rate: float
    normalize: Optional[Callable]
    mse_by_epoch: List[float]

    def __init__(self, layers: List[Dense], learning_rate: float):
        self.layers = layers
        self.learning_rate = learning_rate

        self.normalize = None
        self.mse_by_epoch = []

        layers_count = len(layers)

        if layers_count < MINIMUM_LAYER_COUNT:
            raise ValueError("The minimum allowable number of layers is "
                             f"{MINIMUM_LAYER_COUNT}. You passed on: {layers_count}")
        if self.first_layer.activation_function:
            raise Warning("The input layer does not need the activation function")
        if self.last_layer.use_bias:
            raise Warning("The last layer does not need bias!")
        if not 0 < learning_rate <= 1:
            raise ValueError(f"Learning rate must be within (0, 1] range, not {learning_rate}")

        for layer, next_layer in pairwise(layers):
            layer.initialize_weights(next_layer.neurons)

    @property
    def first_layer(self) -> Dense:
        """
        Input layer
        """
        return self.layers[0]

    @property
    def last_layer(self) -> Dense:
        """
        Output layer
        """
        return self.layers[-1]

    def __query(self, data):
        """
        The transmitted data passes through the entire network to get response
        """
        self.first_layer.output = data

        for layer, next_layer in pairwise(self.layers):
            next_layer.input = layer.weights @ layer.output
            next_layer.output = next_layer.activation_function.f(next_layer.input)

        return self.last_layer.output

    def __backpropagation(self, errors, batch_size: int) -> None:
        """
        Changing weights using backpropagation algorithm

        :param errors: Neural network errors in the last 'batch_size' trainings
        :param batch_size: Batch size
        """
        mean_error = sum(errors) / batch_size
        self.last_layer.error = mean_error

        for next_layer, layer in pairwise(reversed(self.layers)):
            layer.error = layer.weights.T @ next_layer.error
            gradient = (next_layer.error * next_layer.activation_function.df(next_layer.input)) @ layer.output.T
            layer.weights += self.learning_rate * gradient

    def train(self,
              inputs,
              outputs,
              batch_size: int = 10,
              epochs: int = 5,
              shuffle: bool = True,
              normalize: bool = True,
              scope=(0, 1)):
        """
        Neural network training

        :param inputs: List of lists or 2d-numpy-array with input data

        :param outputs: List of lists or 2d-numpy-array with expected output

        :param batch_size: Number of training sessions in one batch

        :param epochs: The number of full passes of the transmitted data through the neural network

        :param shuffle: Whether to shuffle data between epochs

        :param normalize: Whether to normalize the data

        :param scope: Data normalisation range, used only with param normalize=True
        """
        if len(inputs) != len(outputs):
            raise ValueError(f"Inputs len ({len(inputs)}) != Outputs len ({len(outputs)})")

        for index, input_ in enumerate(inputs):
            if len(input_) == self.first_layer.neurons:
                continue
            raise ValueError(f"Input at {index=}: Amount of data ({len(input_)}) != "
                             f"Number of neurons in the first layer ({self.first_layer.neurons}")

        for index, output in enumerate(outputs):
            if len(output) == self.last_layer.neurons:
                continue
            raise ValueError(f"Output at {index=}: Amount of data ({len(output)}) != "
                             f"Number of neurons in the last layer ({self.last_layer.neurons}")

        training_dataset = np.array(inputs)
        expected_outputs = np.array(outputs)

        if normalize:
            min_value = training_dataset.min()
            max_value = training_dataset.max()
            self.normalize = make_normalization_function(min_value, max_value, scope=scope)
            training_dataset = self.normalize(training_dataset)

        for _ in range(epochs):
            if shuffle:
                training_dataset, expected_outputs = shuffle_(training_dataset, expected_outputs)

            batches = make_batches(training_dataset, batch_size), make_batches(expected_outputs, batch_size)

            actual = expected = None
            for batch in zip(*batches):
                errors = []
                for data, expected in zip(*batch):
                    actual = self.__query(to_vector(data))
                    expected = to_vector(expected)
                    errors.append(expected - actual)
                self.__backpropagation(errors, batch_size)
            self.mse_by_epoch.append(mse(expected, actual))

    def predict(self, data) -> List[float]:
        """
        Get a neural network response

        :param data: List with input data
        :return: List with output data
        """
        if len(data) != self.first_layer.neurons:
            raise ValueError(f"Data len ({len(data)}) != "
                             f"Number of neurons in the first layer ({self.first_layer.neurons})")

        data = to_vector(data)

        if self.normalize:
            data = self.normalize(data)

        # The response returns as a two-dimensional numpy array
        response = self.__query(data)
        return response.flatten().tolist()
