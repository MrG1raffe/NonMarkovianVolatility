import numpy as np
from numpy.typing import NDArray
from numpy import float64
from numba import jit
from typing import Callable

from signatures.tensor_sequence import TensorSequence

@jit(nopython=True)
def ode_pece(
    func: Callable,
    t_grid: NDArray[float64],
    u: TensorSequence,
) -> TensorSequence:
    dt = np.diff(t_grid)

    psi = u * 1
    psi_pred = u * 1
    for i in range(len(dt)):
        psi_pred.update(psi + func(psi) * dt[i])
        psi.update(psi + (func(psi_pred) + func(psi)) * (dt[i] / 2))

    return psi

@jit(nopython=True)
def ode_rk4(
    func: Callable,
    t_grid: NDArray[float64],
    u: TensorSequence,
) -> TensorSequence:
    dt = np.diff(t_grid)
    psi = u * 1

    k1 = func(psi) * 0
    k2 = func(psi) * 0
    k3 = func(psi) * 0
    k4 = func(psi) * 0

    for i in range(len(dt)):
        k1.update(func(psi))
        k2.update(func(psi + k1 * (dt[i] / 2)))
        k3.update(func(psi + k2 * (dt[i] / 2)))
        k4.update(func(psi + k3 * dt[i]))
        psi.update(psi + (k1 + k2 * 2 + k3 * 2 + k4) * (dt[i] / 6))

    return psi