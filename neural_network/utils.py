"""
Side functions
"""
import numpy as np
from typing import Callable, Tuple


def convert_to_vector(data):
    """
    Convert data to 2d-numpy-vector

    Example of use
    --------------
    >>> vector = convert_to_vector([1, 2, 3])
    >>> print(vector)
    array([[1],
           [2],
           [3]])
    >>> vector.shape # three rows, one column
    (3, 1)
    """
    return np.array(data, ndmin=2).T


def make_batches(container, batch_size=1):
    """
    Splits the sequence into parts

    Example of use
    --------------
    >>> make_batches([1, 2, 3], 2)
    [[1, 2], [3]]
    """
    return [container[i: i + batch_size] for i in range(0, len(container), batch_size)]


def make_normalization_function(min_value: float, max_value: float, scope=(0, 1)) -> Callable:
    """
    :param min_value: Min value in data
    :param max_value: Max value in data
    :param scope: Normalization range
    :return: normalization function
    """
    def normalize(x):
        return scope[0] + (x - min_value) / (max_value - min_value) * (scope[1] - scope[0])
    return normalize


def mse(expected: np.ndarray, actual: np.ndarray):
    """
    Mean square error
    """
    n = expected.shape[0]

    return (1 / n * sum((expected - actual) ** 2))[0]


def pairwise(iterable):
    """
    To iterate over the current object and the next object in some iterated object

    :param iterable: object that can be iterated over

    Example of use
    --------------
    >>> array = [0, 1, 2]
    >>> for c, d in pairwise(array):
    ...       print(c, d)
    ...
    0 1
    1 2
    """
    it = iter(iterable)
    a = next(it, None)

    for b in it:
        yield a, b
        a = b


def shuffle_(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mixes the two numpy arrays in the same way.

    Their length should be the same

    Example of use
    --------------
    >>> shuffle_(np.array([1, 2, 3, 4]), np.array([4, 3, 2, 1]))
    (array([3, 2, 4, 1]), array([2, 3, 1, 4]))
    """
    indices = np.arange(a.shape[0])
    np.random.shuffle(indices)
    return a[indices], b[indices]
