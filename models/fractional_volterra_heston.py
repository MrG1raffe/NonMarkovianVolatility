import numpy as np
from numpy.typing import NDArray
from math import gamma
from numpy import float_, complex_, complex128
from typing import Union, Tuple
from dataclasses import dataclass
from math import ceil
from numba import jit, prange

from models.characteristic_function_model import CharacteristicFunctionModel
from utility.utility import to_numpy


@dataclass
class FractionalVolterraHeston(CharacteristicFunctionModel):
    """
    A class describing the Lifted Heston model

    dF_t/F_t = sqrt(V_t) * dW_t,
    V_t = V_0 + ∫ K(t - s) * λ * (θ - V_s) ds + ∫ K(t - s) * ν * sqrt(V_s) * dB_s, d<W, B>_t = ρ * dt,
    where K(t) = (t + ε)^{H - 0.5} / Г(H + 0.5) is a shifted fractional kernel.

    theta: see the equation.
    lam: see the equation.
    nu: see the equation.
    rho: correlation between B_t and W_t.
    V0: initial value of variance.
    H: roughness parameter of the kernel
    eps: shift parameter of the kernel.
    """
    theta: float
    lam: float
    nu: float
    rho: float
    H: float
    eps: float
    V0: float

    @staticmethod
    def compile():
        _jit_parallel_char_func(t_grid=np.linspace(0, 1, 3), u_arr=np.zeros(2),
                                nu=1, lam=0, rho=0, theta=0.1, V0=0.1, H=0.5, eps=0)

    def characteristic_function(
        self,
        T: float,
        x: float,
        u1: NDArray[complex_],
        cf_timestep: float = 0.001,
        **kwargs
    ) -> NDArray[complex_]:
        """
        Computes the generalized characteristic function

        E[exp{i * u1 * X_T + i * u2 * V_T + i * f1 * ∫ X_s ds + i * f2 * ∫ V_s ds}]     (1)

        for the given model, where X_t = F_t if `model_type` == "normal" and
        X_t = log(F_t) if `model_type` == "log-normal".

        :param u1: X_T coefficient in the characteristic function, see (1).
        :param T: date in the characteristic function, see (1).
        :param x: X_0, equals to F_0 if `model_type` == "normal" and to log(F_0) if `model_type` == "log-normal".
        :param cf_timestep: a timestep to be used in numerical scheme for the Riccati equation.
        :return: a value of the characteristic function (1) for the given coefficients.
        """
        u_arr = to_numpy(1j * u1)

        u_shape = u_arr.shape
        u_arr = u_arr.flatten()

        timestep = min(cf_timestep, T / 10)
        t_grid = np.linspace(0, T, ceil(T / timestep) + 1)

        res = _jit_parallel_char_func(t_grid=t_grid, u_arr=u_arr, nu=self.nu, lam=self.lam,
                                      rho=self.rho, theta=self.theta, V0=self.V0, H=self.H, eps=self.eps)
        res *= np.exp(u_arr * x)
        return np.reshape(res, u_shape)


@jit(parallel=True, nopython=True)
def _jit_parallel_char_func(
    t_grid: NDArray[float_],
    u_arr: NDArray[complex_],
    nu: float,
    lam: float,
    rho: float,
    theta: float,
    V0: float,
    H: float,
    eps: float
) -> NDArray[complex128]:
    """
    Calculates the characteristic function for a vector 'u_arr' evaluating the function _jit_char_func for each
    u in u_arr in a parallelized loop. Cf the docstring of _jit_char_func.
    """
    cf_arr = np.zeros(len(u_arr), dtype=complex128)
    for i in prange(len(u_arr)):
        cf_arr[i] = _jit_char_func(t_grid=t_grid, u=u_arr[i], nu=nu, lam=lam, rho=rho, theta=theta, V0=V0, H=H, eps=eps)
    return cf_arr


@jit(nopython=True)
def _jit_char_func(
    t_grid: NDArray[float_],
    u: complex,
    nu: float,
    lam: float,
    rho: float,
    theta: float,
    V0: float,
    H: float,
    eps: float
) -> complex:
    """
    Calculates the characteristic function in the Heston model.

    psi is a solution to the Fractional Volterra Riccati equation

    ψ(t) = K * F(ψ)

    solved with the Adams scheme. Here, the function F is __riccati_func.

    :param t_grid: time grid.
    :param u: characteristic function argument.
    :param nu: model parameter.
    :param lam: model parameter.
    :param rho: model parameter.
    :param theta: model parameter.
    :param V0: model parameter.
    :param H: model parameter.
    :param eps: model parameter.
    :return: psi on `t_grid` as an NDArray of the same size.
    """

    psi = __psi_on_grid(t_grid=t_grid, u=u, nu=nu, lam=lam, rho=rho, H=H, eps=eps)
    F = __riccati_func(u, psi, nu=nu, lam=lam, rho=rho)
    g0 = __g0(t_grid, V0, theta=theta, lam=lam, H=H, eps=eps)
    res = np.exp(np.trapz(np.flip(F) * g0, x=t_grid))

    return res


@jit(nopython=True)
def __riccati_func(
    u: complex,
    psi: Union[complex, NDArray[complex_]],
    nu: float,
    lam: float,
    rho: float
):
    """
    The function appearing on the right hand side of the Riccati equation for psi.

    :param u: characteristic function argument.
    :param psi: psi as a number or array.
    :param nu: model parameter.
    :param lam: model parameter.
    :param rho: model parameter.
    :return: the value of the function F on the time grid `t`.
    """
    return 0.5 * (u ** 2 - u) + (u * rho * nu - lam) * psi + 0.5 * (nu * psi) ** 2


@jit(nopython=True)
def __g0(
    t: Union[float, NDArray[float_]],
    V0: float,
    theta: float,
    lam: float,
    H: float,
    eps: float
) -> Union[float, NDArray[float_]]:
    """
    Calculates the function g0.
    """
    return V0 + lam * theta / gamma(H + 1.5) * ((t + eps) ** (H + 0.5) - eps ** (H + 0.5))


@jit(nopython=True)
def __b_coefs(
    t_grid: NDArray[float_],
    k: int,
    H: float,
    eps: float
) -> NDArray[float_]:
    """
    Calculates a vector of 'b_k' coefficients from Adams method
    """
    b_k = 1 / gamma(H + 1.5) * ((t_grid[k + 1] - t_grid[:k + 1] + eps) ** (H + 0.5) -
                                (t_grid[k + 1] - t_grid[1:k + 2] + eps) ** (H + 0.5))
    return b_k


@jit(nopython=True)
def __a_coefs(
    t_grid: NDArray[float_],
    k: int,
    H: float,
    eps: float
) -> Tuple[NDArray[float_], float]:
    """
    Calculates a vector of 'a_k' coefficients from Adams method
    """
    dt = t_grid[1] - t_grid[0]
    a_k = np.zeros(k + 1)
    a_k[0] = 1 / dt / gamma(H + 2.5) * (dt * (H + 1.5) * (t_grid[k + 1] + eps) ** (H + 0.5) +
                                        (t_grid[k + 1] - dt + eps) ** (H + 1.5) -
                                        (t_grid[k + 1] + eps) ** (H + 1.5))
    a_k[1:] = 1 / dt / gamma(H + 2.5) * ((t_grid[k + 1] - t_grid[:k] + eps) ** (H + 1.5) +
                                         (t_grid[k + 1] - t_grid[2:k + 2] + eps) ** (H + 1.5) -
                                         2 * (t_grid[k + 1] - t_grid[1:k + 1] + eps) ** (H + 1.5))
    a_k_last = 1 / dt / gamma(H + 2.5) * (-dt * (H + 1.5) * eps ** (H + 0.5) +
                                          (dt + eps) ** (H + 1.5) - eps ** (H + 1.5))
    return a_k, a_k_last


@jit(nopython=True)
def __psi_on_grid(
    t_grid: NDArray[float_],
    u: complex,
    nu: float,
    lam: float,
    rho: float,
    H: float,
    eps: float
) -> NDArray[complex_]:
    """
    Solves the Riccati equation via the Adams scheme.
    """
    psi = np.zeros(len(t_grid), dtype=complex128)
    F = np.zeros(len(t_grid), dtype=complex128)
    for k in range(len(psi) - 1):
        F[k] = __riccati_func(u, psi[k], nu=nu, lam=lam, rho=rho)

        b_k = __b_coefs(t_grid, k, H=H, eps=eps)
        psi_pred = b_k.astype(complex128) @ F[:k + 1]

        a_k, a_k_last = __a_coefs(t_grid, k, H=H, eps=eps)
        psi[k + 1] = a_k.astype(complex128) @ F[:k + 1] + a_k_last * __riccati_func(u=u, psi=psi_pred, nu=nu,
                                                                                    lam=lam, rho=rho)
    return psi
