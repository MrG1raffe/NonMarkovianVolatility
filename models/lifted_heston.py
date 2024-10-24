import numpy as np
from numpy.typing import NDArray
from numpy import float_, complex_, complex128
from typing import Union, Tuple
from dataclasses import dataclass
from math import ceil, gamma
from numba import jit, prange

from models.heston import Heston, _jit_heston_riccati_func
from simulation.diffusion import Diffusion
from utility.utility import DEFAULT_SEED, to_numpy


@dataclass
class LiftedHeston(Heston):
    """
    A class describing the Lifted Heston model

    dF_t /F_t = sqrt(V_t)* dW_t,
    V_t = V_0 + ∫ Σ c_i * exp{-x_i * (t - s)} * (λ * (θ - V_s) ds + ν * sqrt(V_s) * dB_s).

    theta: see the equation.
    lam: see the equation.
    nu: see the equation.
    rho: spot-vol correlation.
    x: mean-reversion coefficients in the kernel, see the equation.
    c: coefficients in the kernel, see the equation.
    V0: initial value of variance.
    """
    theta: Union[float, NDArray[float_]]
    lam: float
    nu: float
    rho: float
    x: NDArray[float_]
    c: NDArray[float_]
    V0: float
    H: float = None
    r: float = None
    n: int = None

    def __post_init__(self):
        if self.x is None or self.c is None:
            # Initialize c and x using the equation (3.3) for "Lifring the Heston model"
            if self.H is None or self.r is None or self.n is None:
                raise ValueError("Not enough parameters were given to initialize x and c.")
            a, r, n = self.H + 0.5, self.r, self.n
            ii = np.arange(1, n + 1)
            self.c = (r**(1 - a) - 1) * r**((a - 1) * (1 + n / 2)) / gamma(a) / gamma(2 - a) * r**((1 - a) * ii)
            self.x = (1 - a) / (2 - a) * (r**(2 - a) - 1) / (r**(1 - a) - 1) * r**(ii - 1 - n / 2)

        # convert everything to numpy
        self.x = np.array(self.x)
        self.c = np.array(self.c)

    @staticmethod
    def compile():
        t_grid = np.linspace(0, 1, 100)
        _jit_parallel_char_func(t_grid=t_grid, u_arr=np.array([0, 0.5]),
                                nu=1, lam=0.1, rho=-0.1, theta=0.1, V0=0.1, g0_arr=np.zeros_like(t_grid))

    def g0(
        self,
        t: Union[float, NDArray[float_]]
    ) -> NDArray[float_]:

        """
        V_t = V_0 + ∫ Σ c_i * exp{-x_i * (t - s)} * (λ * (θ - V_s) ds + ν * sqrt(V_s) * dB_s).
        Calculates the function

        g_0(t) = V_0 + λ * θ * Σ c_i * (1 - exp{-x_i * t}) / x_i.

        :param t: value of t, number or 1D array.
        :return: g0(t) for the given `t`.
        """
        return self.V0 + self.lam * self.theta * np.sum(
            np.divide(self.c[None, :], self.x[None, :], where=~np.isclose(self.x[None, :], 0)) *
            (1 - np.exp(-self.x[None, :] * np.reshape(t, (-1, 1)))) +
            self.c[None, :] * np.reshape(t, (-1, 1)) * np.isclose(self.x[None, :], 0),
            axis=1
        )

    def characteristic_function(
        self,
        T: float,
        x: float,
        u1: NDArray[complex_],
        cf_timestep: float = 0.001,
        scheme: str = "exp",
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

        g0_arr = self.g0(t_grid)

        res = _jit_parallel_char_func(t_grid=t_grid, u_arr=u_arr, c=self.c, x=self.x, nu=self.nu, lam=self.lam,
                                      rho=self.rho, g0_arr=g0_arr, scheme=scheme)
        res *= np.exp(u_arr * x)
        return np.reshape(res, u_shape)

    def get_variance_trajectory(
            self,
            t_grid: NDArray[float_],
            size: int,
            return_factors: bool = False,
            rng: np.random.Generator = None,
            scheme: str = "exp",
            B_traj: NDArray[float_] = None
    ) -> Union[NDArray[float_], Tuple[NDArray[float_], NDArray[float_]]]:
        """
        Simulate the variance and the factor processes on the given time grid.

        :param t_grid: time gird.
        :param size: number of trajectories to simulate.
        :param return_factors: whether to return the factors U_i together with the variance.
        :param rng: random number generator to simulate the trajectories with.
        :param scheme: a discretization Monte Carlo scheme, either "exp", or "semi-implicit".
        :param B_traj: pre-simulated trajectories of the BM B_t corresponding to the stochastic volatility.
            By default, None, and will be simulated within the function.
        :return: an array of shape (size, len(t_grid)) for the variance and an array of shape
            (size, n_stoch_factors, len(t_grid)) for U if `return_factors` == True.
        """
        if rng is None:
            rng = np.random.default_rng(seed=DEFAULT_SEED)

        if B_traj is None:
            # simulation of B_traj
            diffusion = Diffusion(t_grid=t_grid, dim=1, size=size, rng=rng)
            B_traj = diffusion.brownian_motion()[:, 0, :]  # shape (size, len(t_grid))
        else:
            if B_traj.shape != (size, len(t_grid)):
                raise ValueError("Inconsistent dimensions of B_traj were given.")

        # simulation of U
        U_traj = np.zeros((size, len(self.x), len(t_grid)))
        V_traj = np.zeros((size, len(t_grid)))
        V_traj += self.g0(t_grid)  # V_traj[:, 0] is automatically set equal to V0
        dB_traj = np.diff(B_traj, axis=1)
        dt = np.diff(t_grid)
        if np.any(dt < 0):
            raise ValueError("Time grid should be increasing.")
        if scheme == "semi-implicit":
            scale = 1 / (1 + np.reshape(self.x, (1, -1)) * dt[:, None])
        elif scheme == "exp":
            scale = np.exp(-np.reshape(self.x, (1, -1)) * dt[:, None])
        else:
            raise ValueError("Incorrect value of `scheme` was given.")
        for k in range(len(t_grid) - 1):
            V_k = V_traj[:, k][:, None]
            U_traj[:, :, k + 1] = scale[k] * (U_traj[:, :, k] - self.lam * dt[k] * V_k +
                                              self.nu * np.sqrt(np.maximum(V_k, 0)) * dB_traj[:, k][:, None])

            V_traj[:, k + 1] += U_traj[:, :, k + 1] @ self.c
            V_traj[:, k + 1] = np.maximum(0, V_traj[:, k + 1])

        if return_factors:
            return V_traj, U_traj
        else:
            return V_traj


@jit(parallel=True, nopython=True)
def _jit_parallel_char_func(
    t_grid: NDArray[float_],
    u_arr: NDArray[complex_],
    scheme: str,
    c: NDArray[float_],
    x: NDArray[float_],
    nu: float,
    lam: float,
    rho: float,
    g0_arr: NDArray[float_]
) -> NDArray[complex128]:
    cf_arr = np.zeros(len(u_arr), dtype=complex128)
    for i in prange(len(u_arr)):
        cf_arr[i] = _jit_char_func(t_grid=t_grid, u=u_arr[i], c=c, x=x, scheme=scheme, nu=nu, lam=lam, rho=rho, g0_arr=g0_arr)
    return cf_arr


@jit(nopython=True)
def _jit_char_func(
    t_grid: NDArray[float_],
    u: complex,
    c: NDArray[float_],
    x: NDArray[float_],
    nu: float,
    lam: float,
    rho: float,
    g0_arr: NDArray[float_],
    scheme: str = "exp",
) -> complex:
    """
    Calculates the characteristic function in the Heston model.

    psi is a solution to the vector Riccati equation

    (ψ^i)'(t) = -x_i * ψ^i(t) + F(u, Σ c_j * ψ^j(t)),
    (ψ^i)'(0) = 0,

    where the function F is `_jit_heston_riccati_func`.

    :param t_grid: time grid.
    :param u: characteristic function argument.
    :param c: model parameters.
    :param x: model parameters.
    :param nu: model parameter.
    :param lam: model parameter.
    :param rho: model parameter.
    :param scheme: numerical scheme for vector Riccati.
    :return: psi on `t_grid` as an NDArray of the same size.
    """

    psi = np.zeros((t_grid.size, x.size), dtype=np.complex128)
    dt = np.reshape(np.diff(t_grid), (-1, 1))
    x_row = np.reshape(x, (1, -1))
    if np.any(dt < 0):
        raise ValueError("Time grid should be increasing.")
    if scheme == "semi-implicit":
        scale_1 = 1 / (1 + x_row * dt)
        scale_2 = scale_1 * dt
    elif scheme == "exp":
        EPS = 1e-6
        scale_1 = np.exp(-x_row * dt)
        if np.min(np.abs(x_row)) < EPS:
            scale_2 = scale_1 * dt  # To doublecheck: why applied to all x
        else:
            scale_2 = (1 - scale_1) / x_row
    else:
        raise ValueError("Incorrect value of `scheme` was given.")

    F = np.zeros(t_grid.size, dtype=complex128)
    for i in range(len(t_grid) - 1):
        psi_i = np.sum(psi[i] * c)
        F[i] = _jit_heston_riccati_func(u=u, psi=psi_i, nu=nu, lam=lam, rho=rho)
        psi[i + 1] = scale_1[i] * psi[i] + scale_2[i] * F[i]
    F[-1] = _jit_heston_riccati_func(u=u, psi=np.sum(psi[-1] * c), nu=nu, lam=lam, rho=rho)

    return np.exp(np.trapz(np.flip(F) * g0_arr, x=t_grid))
