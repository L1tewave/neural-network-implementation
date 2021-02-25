"""
Here, the definition of activation functions
"""
from __future__ import annotations
from enum import Enum
from numpy import maximum
from scipy.special import expit
from typing import Callable, Optional


# The functions are used with numpy arrays
def _relu_function(x):
    return maximum(x, 0)


def _relu_derivative(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


def _sigmoid_function(x):
    return expit(x)


def _sigmoid_derivative(x):
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
