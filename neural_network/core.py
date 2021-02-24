"""
The core contains the definition of the perceptron class and the layer class
"""
from __future__ import annotations
import numpy as np
from typing import Callable, List, Union, Optional

from neural_network.activation_functions import ActivationFunction
from neural_network.utils import convert_to_vector, is_empty, make_normalization_function, pairwise, shuffle_


class Dense:
    """
    Fully-connected layer of neurons
    """

    def __init__(self, neurons: int, activation_function: Union[str, ActivationFunction, None] = None,
                 use_bias: bool = False):
        if any([neurons < 0, not isinstance(neurons, int)]):
            raise ValueError("The number of neurons must be a positive integer")
        self.__neurons = neurons

        if activation_function is None:
            self.__activation_function = None
        elif isinstance(activation_function, ActivationFunction):
            self.__activation_function = activation_function
        elif isinstance(activation_function, str):
            self.__activation_function = ActivationFunction.get_by_name(activation_function)
        else:
            raise ValueError(f"This type: {activation_function} of activation function is not available")

        self.use_bias = use_bias
        self.weights = None

        self.input = None
        self.__output = None
        self.__error = None

    def initialize_weights(self, next_layer_neurons: int) -> None:
        matrix_size = (next_layer_neurons, self.neurons + 1) if self.use_bias else (next_layer_neurons, self.neurons)
        self.weights = np.random.normal(loc=0.0, scale=pow(matrix_size[1], -0.5), size=matrix_size)

    def is_first(self) -> bool:
        return self.activation_function is None

    def is_last(self) -> bool:
        return self.weights is None

    @property
    def neurons(self) -> int:
        return self.__neurons

    @property
    def activation_function(self) -> ActivationFunction:
        return self.__activation_function

    @property
    def output(self) -> np.ndarray:
        return self.__output

    @output.setter
    def output(self, value):
        if self.use_bias:
            self.__output = np.vstack((value, 1))
        else:
            self.__output = value

    @property
    def error(self) -> np.ndarray:
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


# Some constants for neural networks
MINIMUM_LAYER_COUNT = 2
FIRST_LAYER = 0
LAST_LAYER = -1


class Perceptron:
    """
    Multilayer perceptron class
    """

    def __init__(self, layers: List[Dense], learning_rate: float):

        if len(layers) < MINIMUM_LAYER_COUNT:
            raise ValueError("The minimum allowable number of layers is "
                             f"{MINIMUM_LAYER_COUNT}. You passed on: {len(layers)}")
        if layers[LAST_LAYER].use_bias is True:
            raise Warning("The last layer does not need bias")
        if not 0 < learning_rate <= 1:
            raise ValueError(f"Learning rate must be in must be within (0, 1], not {learning_rate}")

        self.learning_rate = learning_rate

        for layer, next_layer in pairwise(layers):
            layer.initialize_weights(next_layer.neurons)

        self.layers = layers
        self.layers_count = len(layers)

        self.normalize: Optional[Callable] = None

    def __query(self, data) -> np.ndarray:
        """
        The transmitted data passes through the entire network to get response.
        """
        self.layers[FIRST_LAYER].output = convert_to_vector(data)

        for layer, next_layer in pairwise(self.layers):
            next_layer.input = layer.weights @ layer.output
            next_layer.output = next_layer.activation_function.f(next_layer.input)

        return self.layers[LAST_LAYER].output

    def __backpropagation(self, errors, batch_size: int) -> None:
        """
        Changing weights using backpropagation algorithm

        :param errors: Neural network errors in the last 'batch_size' trainings
        :param batch_size: Batch size
        """
        mean_error = sum(errors) / batch_size
        self.layers[LAST_LAYER].error = np.array(mean_error, ndmin=2)

        for next_layer, layer in pairwise(reversed(self.layers)):
            layer.error = layer.weights.T @ next_layer.error
            gradient = (next_layer.error * next_layer.activation_function.df(next_layer.input)) @ layer.output.T
            layer.weights += self.learning_rate * gradient

    def train(self, inputs: List[List[int]], outputs: List[List[int]], batch_size: int = 10, epochs: int = 5,
              shuffle: bool = False, normalize: bool = True) -> None:
        """
        Neural network training.

        :param inputs: List of lists with input data

        :param outputs: List of lists with the expected output

        :param batch_size: Number of training sessions in one batch

        :param epochs: The number of full passes of the transmitted data through the neural network

        :param shuffle: Whether to shuffle data between epochs

        :param normalize: Whether to normalize the data
        """
        if len(inputs) != len(outputs):
            raise ValueError(f"The input data length {len(inputs)} must be "
                             f"equal to the output data length {len(outputs)} !")

        for index, input_ in enumerate(inputs):
            if len(input_) == self.layers[FIRST_LAYER].neurons:
                continue
            raise ValueError(f"Input data with index {index} contain not enough data!")

        for index, output in enumerate(outputs):
            if len(output) == self.layers[LAST_LAYER].neurons:
                continue
            raise ValueError(f"Output data with index {index} contain not enough data!")

        training_dataset = np.array(inputs)
        training_quantity = len(training_dataset)

        if normalize is True:
            min_value, max_value = np.amin(training_dataset), np.amax(training_dataset)
            self.normalize = make_normalization_function(min_value, max_value, scope=(-1, 1))
            training_dataset = self.normalize(training_dataset)

        expected_outputs = np.array(outputs)

        errors = []
        for _ in range(epochs):
            if not is_empty(errors):
                self.__backpropagation(errors, batch_size)
                errors = []

            if shuffle is True:
                training_dataset, expected_outputs = shuffle_(training_dataset, expected_outputs)

            for index, data, expected in zip(range(1, training_quantity + 1), training_dataset, expected_outputs):

                expected = convert_to_vector(expected)
                actual = self.__query(data)
                errors.append(expected - actual)

                if index % batch_size == 0:
                    self.__backpropagation(errors, batch_size)
                    errors = []

    def predict(self, data: List[int]) -> List[int]:
        """
        Get a neural network response

        :param data: List with input data
        :return: List with output data
        """
        if len(data) != self.layers[FIRST_LAYER].neurons:
            raise ValueError(f"Input data contain not enough data to predict (less than input neurons count)!")
        if self.normalize is not None:
            data = self.normalize(data)

        response = self.__query(data)
        return response.reshape(response.shape[1]).tolist()
