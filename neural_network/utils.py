"""
Side functions
"""
import numpy as np
from typing import List, Tuple, Union


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


def is_empty(container):
    return len(container) == 0


def make_normalization_function(min_value: float, max_value: float, scope: Tuple[int, int] = (0, 1)):
    def normalize(x):
        return scope[0] + (x - min_value) / (max_value - min_value) * (scope[1] - scope[0])

    return normalize


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


def shuffle_(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mixes the two numpy arrays in the same way

    Their length should be the same
    """
    indices = np.arange(a.shape[0])
    np.random.shuffle(indices)
    return a[indices], b[indices]
