from __future__ import annotations
from enum import Enum
from typing import Callable, Optional, List, Union, Tuple
import numpy as np
from scipy.special import expit


def _relu_function(x: np.ndarray):
    return np.maximum(x, 0)


def _relu_derivative(x: np.ndarray):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


def _sigmoid_function(x: np.ndarray):
    return expit(x)


def _sigmoid_derivative(x: np.ndarray):
    f = _sigmoid_function
    return f(x) * (1 - f(x))


class ActivationFunction(Enum):
    """
    A enumeration of activation functions. Each element contains the function and the derivative of the function
    """
    ReLu = ("Rectified linear unit", _relu_function, _relu_derivative)
    Sigmoid = ("Soft step", _sigmoid_function, _sigmoid_derivative)

    def __init__(self, nickname: str, f: Callable, df: Callable) -> None:
        self.nickname = nickname
        self.f = f
        self.df = df

    @staticmethod
    def get_by_name(name: str) -> Optional[ActivationFunction]:
        return next((af for af in ActivationFunction if af.name.lower() == name.lower()), None)


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
        self.error = None

    def initialize_weights(self, next_layer_neurons: int) -> None:
        matrix_size = (next_layer_neurons, self.neurons + 1) if self.use_bias else (next_layer_neurons, self.neurons)
        self.weights = np.random.normal(loc=0.0, scale=pow(matrix_size[1], -0.5), size=matrix_size)

    def calculate_next_layer_input(self):
        return self.weights @ self.output

    @property
    def neurons(self) -> int:
        return self.__neurons

    @property
    def activation_function(self) -> ActivationFunction:
        return self.__activation_function

    @property
    def output(self):
        return self.__output

    @output.setter
    def output(self, value):
        if self.use_bias:
            self.__output = np.vstack((value, 1))
        else:
            self.__output = value

    def __repr__(self):
        return f"Fully-connected layer [" \
               f"neurons = {self.neurons}, " \
               f"use_bias = {self.use_bias}, " \
               f"First layer? = {True if self.input is None else False}, " \
               f"Last layer? = {True if self.weights is None else False}, " \
               f"activation function = {self.activation_function.name}" \
               f"]"


def make_normalization_function(min_value: float, max_value: float, scope: Tuple[int, int] = (0, 1)):
    def normalize(x):
        return scope[0] + (x - min_value) / (max_value - min_value) * (scope[1] - scope[0])

    return normalize


def convert_to_vector(data: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Convert data to vector

    Example of use
    -------------
    >>> some_data = [0, 1, 2, 3]
    >>> vector = convert_to_vector(some_data)
    >>> vector.shape
    (4, 1)
    """
    return np.array(data, ndmin=2).T


def _shuffle(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mixes the two numpy arrays in the same way

    Their length should be the same
    """
    indices = np.arange(a.shape[0])
    np.random.shuffle(indices)
    return a[indices], b[indices]


def empty(container):
    return len(container) == 0


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
        if not 0 < learning_rate <= 1:
            raise ValueError(f"Learning rate must be in must be within (0, 1], not {learning_rate}")

        self.learning_rate = learning_rate

        self.layers_count = len(layers)
        self.layers = layers
        self.reversed_layers = layers[::-1]

        for layer, next_layer in pairwise(layers):
            layer.initialize_weights(next_layer.neurons)

        self.normalize = None

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

        training_quantity = len(inputs)
        training_dataset = np.array(inputs)

        if normalize is True:
            min_value, max_value = np.amin(training_dataset), np.amax(training_dataset)
            self.normalize = make_normalization_function(min_value, max_value, scope=(-1, 1))
            training_dataset = self.normalize(training_dataset)

        expected_outputs = np.array(outputs)

        errors = []
        for _ in range(epochs):
            if not empty(errors):
                self.__backpropagation(errors, batch_size)
                errors = []

            if shuffle is True:
                training_dataset, expected_outputs = _shuffle(training_dataset, expected_outputs)

            for index, data, expected in zip(range(1, training_quantity + 1), training_dataset, expected_outputs):
                expected = convert_to_vector(expected)
                actual = self.__query(data)
                errors.append(expected - actual)
                if index % batch_size == 0:
                    self.__backpropagation(errors, batch_size)
                    errors = []

    def __query(self, data):
        """
        The transmitted data passes through the entire network to get response.
        """
        self.layers[FIRST_LAYER].output = convert_to_vector(data)

        for layer, next_layer in pairwise(self.layers):
            next_layer.input = layer.calculate_next_layer_input()
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


def pairwise(iterable):
    """
    To iterate over the current object and the next object in some iterated object

    :param iterable: object that can be iterated over
    """
    it = iter(iterable)
    a = next(it, None)

    for b in it:
        yield a, b
        a = b
