"""
Here, the definition of activation functions

Functions must be used with numpy arrays!
"""
from __future__ import annotations

from enum import Enum
from typing import Callable
from typing import Optional

import numpy as np


def _relu_function(x):
    return np.maximum(x, 0)


def _relu_derivative(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


def _sigmoid_function(x):
    return 1 / (1 + np.exp(-x))


def _sigmoid_derivative(x):
    f = _sigmoid_function
    return f(x) * (1 - f(x))


def _softplus_function(x):
    return np.log(1 + np.power(np.e, x))


_softplus_derivative = _sigmoid_function


class ActivationFunction(Enum):
    """
    A enumeration of activation functions. Each element contains the function and the derivative of the function
    """
    ReLu = ("Rectified linear unit", _relu_function, _relu_derivative)
    Sigmoid = ("Soft step", _sigmoid_function, _sigmoid_derivative)
    SoftPlus = ("Softplus", _softplus_function, _softplus_derivative)

    def __init__(self, nickname: str, f: Callable, df: Callable):
        self.nickname = nickname
        self.f = f
        self.df = df

    @classmethod
    def get_by_name(cls, name: str) -> Optional[ActivationFunction]:
        """
        Get ActivationFunction object by name or None, if there is no function with this name
        """
        activation_function = next((f for f in cls if f.name.lower() == name.lower()), None)
        if not activation_function:
            raise ValueError(f"Non-existing function name: {name}")
        return activation_function
