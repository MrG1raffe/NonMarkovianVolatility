import numpy as np
from numpy.typing import NDArray
from numpy import float64
from numba import jit
from typing import Callable

from signatures.tensor_sequence import TensorSequence
from signatures.stationary_signature import G, semi_integrated_scheme, discount_ts

@jit(nopython=True)
def func_psi(psi: TensorSequence):
    return psi.proj("2").shuffle_pow(2) / 2 + psi.proj("22") / 2 + psi.proj("1")

@jit(nopython=True)
def func_psi_stat(psi: TensorSequence, lam: float):
    return func_psi(psi) - G(psi) * lam

@jit(nopython=True)
def func_xi(xi: TensorSequence):
    return xi.proj("22") / 2 + xi.proj("1")

@jit(nopython=True)
def psi_riccati_euler(
    t_grid: NDArray[float64],
    u: TensorSequence,
) -> TensorSequence:
    res = np.zeros(t_grid.size, dtype=float64)
    dt = np.diff(t_grid)

    psi = u * 1
    res[0] = np.real(psi[""][0][0])
    for i in range(len(dt)):
        psi.update(psi + func_psi(psi) * dt[i])
        res[i + 1] = np.real(psi[""][0][0])

    return psi, res

@jit(nopython=True)
def psi_riccati_euler_stat(
    t_grid: NDArray[float64],
    u: TensorSequence,
    lam: float
) -> TensorSequence:
    res = np.zeros(t_grid.size, dtype=float64)
    dt = np.diff(t_grid)

    psi = u * 1
    res[0] = np.real(psi[""][0][0])
    for i in range(len(dt)):
        psi.update(discount_ts(ts=psi, dt=dt[i], lam=lam) + semi_integrated_scheme(ts=func_psi(psi), dt=dt[i], lam=lam))
        res[i + 1] = np.real(psi[""][0][0])

    return psi, res

@jit(nopython=True)
def psi_riccati_pece(
    t_grid: NDArray[float64],
    u: TensorSequence,
) -> TensorSequence:
    res = np.zeros(t_grid.size, dtype=float64)
    dt = np.diff(t_grid)

    psi = u * 1
    psi_pred = u * 1
    res[0] = np.real(psi[""][0][0])
    for i in range(len(dt)):
        psi_pred.update(psi + func_psi(psi) * dt[i])
        psi.update(psi + (func_psi(psi_pred) + func_psi(psi)) * (dt[i] / 2))
        res[i + 1] = np.real(psi[""][0][0])

    return psi, res

@jit(nopython=True)
def psi_riccati_pece_stat(
    func: Callable,
    t_grid: NDArray[float64],
    u: TensorSequence,
    lam: float
) -> TensorSequence:
    res = np.zeros(t_grid.size, dtype=float64)
    dt = t_grid[1:] - t_grid[:-1]

    psi = u * 1
    psi_pred = u * 1
    res[0] = np.real(psi[""][0][0])
    for i in range(len(dt)):
        psi_pred.update(discount_ts(ts=psi, dt=dt[i], lam=lam) + semi_integrated_scheme(ts=func(psi), dt=dt[i], lam=lam))
        psi.update(discount_ts(ts=psi, dt=dt[i], lam=lam) + semi_integrated_scheme(ts=(func(psi_pred) + func(psi)) * 0.5, dt=dt[i], lam=lam))
        res[i + 1] = np.real(psi[""][0][0])

    return psi, res

@jit(nopython=True)
def psi_riccati_pecece(
    t_grid: NDArray[float64],
    u: TensorSequence,
) -> TensorSequence:
    res = np.zeros(t_grid.size, dtype=float64)
    dt = np.diff(t_grid)

    psi = u * 1
    psi_pred = u * 1
    psi_pred_pred = u * 1
    res[0] = np.real(psi[""][0][0])
    for i in range(len(dt)):
        psi_pred.update(psi + func_psi(psi) * dt[i])
        psi_pred_pred.update(psi + (func_psi(psi_pred) + func_psi(psi)) * (dt[i] / 2))
        psi.update(psi + (func_psi(psi_pred_pred) + func_psi(psi)) * (dt[i] / 2))
        res[i + 1] = np.real(psi[""][0][0])

    return psi, res

@jit(nopython=True)
def psi_riccati_multistep(
    t_grid: NDArray[float64],
    u: TensorSequence,
) -> TensorSequence:
    res = np.zeros(t_grid.size, dtype=float64)
    dt = np.diff(t_grid)

    psi = u + func_psi(u) * dt[0]
    psi_prev = u * 1
    psi_swap = u * 1
    res[0] = np.real(psi[""][0][0])
    for i in range(len(dt)):
        psi_swap.update(psi)
        psi.update(psi + func_psi(psi) * (1.5 * dt[i]) - func_psi(psi_prev) * (0.5 * dt[i]))
        psi_prev.update(psi_swap)
        res[i + 1] = np.real(psi[""][0][0])

    return psi, res

@jit(nopython=True)
def psi_riccati_rk4(
    t_grid: NDArray[float64],
    u: TensorSequence,
) -> TensorSequence:
    res = np.zeros(t_grid.size, dtype=float64)
    dt = np.diff(t_grid)

    psi = u * 1
    res[0] = np.real(psi[""][0][0])

    k1 = func_psi(psi) * 0
    k2 = func_psi(psi) * 0
    k3 = func_psi(psi) * 0
    k4 = func_psi(psi) * 0

    for i in range(len(dt)):
        k1.update(func_psi(psi))
        k2.update(func_psi(psi + k1 * (dt[i] / 2)))
        k3.update(func_psi(psi + k2 * (dt[i] / 2)))
        k4.update(func_psi(psi + k3 * dt[i]))
        psi.update(psi + (k1 + k2 * 2 + k3 * 2 + k4) * (dt[i] / 6))
        res[i + 1] = np.real(psi[""][0][0])

    return psi, res

@jit(nopython=True)
def xi_riccati(
    t_grid: NDArray[float64],
    xi_0: TensorSequence,
) -> TensorSequence:
    dt = np.diff(t_grid)

    xi = xi_0 * 1
    for i in range(len(dt)):
        xi.update(xi + func_xi(xi) * dt[i])

    return xi