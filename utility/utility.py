import numpy as np
import pandas as pd
from typing import Any, List
from numpy.typing import NDArray
from numpy import float_
from scipy.stats import norm


DEFAULT_SEED = 42


def is_number(x: Any) -> bool:
    """
    Checks whether x is int or float.
    """
    return isinstance(x, int) or isinstance(x, float) or isinstance(x, complex)


def to_numpy(x: Any) -> NDArray[float_]:
    """
    Converts x to numpy array.
    """
    if is_number(x):
        return np.array([x])
    else:
        return np.array(x)


def is_call(flag: str) -> bool:
    """
    Checks whether the flag corresponds to vanilla call option.

    Args:
        flag: flag value.

    Returns:
         True if flag corresponds to call.
    """
    return flag in ["Call", "call", "C", "c"]


def is_put(flag: str) -> bool:
    """
    Checks whether the flag corresponds to vanilla put option.

    Args:
        flag: flag value.

    Returns:
         True if flag corresponds to put.
    """
    return flag in ["Put", "put", "P", "p"]


def from_delta_call_to_strike(
    deltas: NDArray[float_],
    F0: NDArray[float_],
    sigma: NDArray[float_],
    ttm: NDArray[float_]
) -> NDArray[float_]:
    """
    Transforms the delta-strikes in the absolute strikes.

    :param deltas: array of delta-strikes.
    :param F0: underlying price at t = 0.
    :param sigma: array of the implied volatilities corresponding to `K_deltas`.
    :param ttm: array of times to maturity corresponding to `K_deltas`.
    :return: an array of absolute strikes corresponding to `K_deltas`.
    """
    return F0 * np.exp(0.5 * sigma**2 * ttm - sigma * np.sqrt(ttm) * norm.ppf(deltas))
