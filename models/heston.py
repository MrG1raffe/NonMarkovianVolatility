import numpy as np
from numpy.typing import NDArray
from numpy import float_, complex_, complex128
from typing import Union, Tuple
from dataclasses import dataclass
from math import ceil
from numba import jit, prange

from models.characteristic_function_model import CharacteristicFunctionModel
from models.monte_carlo_model import MonteCarloModel
from simulation.diffusion import Diffusion
from utility.utility import DEFAULT_SEED, to_numpy


@dataclass
class Heston(CharacteristicFunctionModel, MonteCarloModel):
    """
    A class describing the Heston model

    dF_t/F_t = sqrt(V_t) * dW_t,
    V_t = V_0 + ∫ λ * (θ - V_s) ds + ∫ ν * sqrt(V_s) * dB_s, d<W, B>_t = ρ * dt.

    theta: see the equation.
    lam: see the equation.
    nu: see the equation.
    rho: correlation between B_t and W_t.
    V0: initial value of variance.
    """
    theta: float
    lam: float
    nu: float
    rho: float
    V0: float

    @staticmethod
    def compile():
        _jit_parallel_char_func(t_grid=np.linspace(0, 1, 100), u_arr=np.array([0, 0.5]),
                                nu=1, lam=0.1, rho=-0.1, theta=0.1, V0=0.1)

    def g0(
        self,
        t: Union[float, NDArray[float_]]
    ) -> NDArray[float_]:

        """
        V_t = V_0 + ∫ Σ c_i * exp{-x_i * (t - s)} * (λ * (θ - V_s) ds + ν * sqrt(V_s) * dB_s).
        Calculates the function

        g_0(t) = V_0 + λ * θ * t.

        :param t: value of t, number or 1D array.
        :return: g0(t) for the given `t`.
        """
        return self.V0 + self.lam * self.theta * t

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
                                      rho=self.rho, theta=self.theta, V0=self.V0)
        res *= np.exp(u_arr * x)
        return np.reshape(res, u_shape)

    def get_variance_trajectory(
        self,
        t_grid: NDArray[float_],
        size: int,
        rng: np.random.Generator = None,
        B_traj: NDArray[float_] = None
    ) -> Union[NDArray[float_], Tuple[NDArray[float_], NDArray[float_]]]:
        """
        Simulate the variance and the factor processes on the given time grid.

        :param t_grid: time gird.
        :param size: number of trajectories to simulate.
        :param rng: random number generator to simulate the trajectories with.
        :param B_traj: pre-simulated trajectories of the BM B_t corresponding to the stochastic volatility.
            By default, None, and will be simulated within the function.
        :return: an array of shape (size, len(t_grid)) for the variance and an array of shape
            (size, 1, len(t_grid)) for U if `return_factors` == True.
        """
        if B_traj is None:
            # simulation of B_traj
            diffusion = Diffusion(t_grid=t_grid, dim=1, size=size, rng=rng)
            B_traj = diffusion.brownian_motion()[:, 0, :]  # shape (size, len(t_grid))
        else:
            if B_traj.shape != (size, len(t_grid)):
                raise ValueError("Inconsistent dimensions of B_traj were given.")

        # simulation of U
        V_traj = np.zeros((size, len(t_grid)))
        V_traj[:, 0] = self.V0
        dB_traj = np.diff(B_traj, axis=1)
        dt = np.diff(t_grid)
        if np.any(dt < 0):
            raise ValueError("Time grid should be increasing.")

        for k in range(len(t_grid) - 1):
            V_k = V_traj[:, k]
            V_traj[:, k + 1] = V_k + self.lam * (self.theta - V_k) * dt[k] + self.nu * np.sqrt(np.maximum(V_k, 0)) * dB_traj[:, k]
            V_traj[:, k + 1] = np.maximum(0, V_traj[:, k + 1])

        return V_traj

    def get_price_trajectory(
        self,
        t_grid: NDArray[float_],
        size: int,
        F0: float,
        rng: np.random.Generator = None,
        return_variance: bool = False,
        **kwargs
    ) -> Union[NDArray[float_], Tuple[NDArray[float_], ...]]:
        """
        Simulates the underlying price trajectories on the given time grid.

        :param t_grid: time grid.
        :param size: number of trajectories to simulate.
        :param F0: initial value of the underlying price.
        :param rng: random number generator to simulate the trajectories with.
        :param return_variance: whether to return the variance V together with the prices.
        :return: an array `F_traj` of shape (size, len(t_grid)) of simulated price trajectories,
            an array `V_traj` of shape (size, len(t_grid)) of variance trajectories if `return_variance` == True,
        """

        if rng is None:
            rng = np.random.default_rng(seed=DEFAULT_SEED)

        diffusion = Diffusion(t_grid=t_grid, dim=2, size=size, rng=rng)

        corr_mat = np.array([[1, self.rho],
                             [self.rho, 1]])
        brownian_motion = diffusion.brownian_motion(correlation=corr_mat)

        dt = np.diff(t_grid)
        dW_traj = np.diff(brownian_motion[:, 0, :], axis=1)  # shape (size, n_hist_factors, len(t_grid)-1)
        B_traj = brownian_motion[:, 1, :]  # shape (size, len(t_grid))

        V_traj = self.get_variance_trajectory(t_grid=t_grid, size=size, B_traj=B_traj)

        log_F_traj = np.log(F0) * np.ones((size, len(t_grid)))
        log_F_traj[:, 1:] += np.cumsum(dW_traj * np.sqrt(V_traj[:, :-1]) - 0.5 * dt * V_traj[:, :-1], axis=1)
        F_traj = np.exp(log_F_traj)

        if return_variance:
            return F_traj, V_traj
        else:
            return F_traj


@jit(nopython=True)
def _jit_heston_riccati_func(
    u: Union[complex, NDArray[complex_]],
    psi: NDArray[complex_],
    nu: float,
    lam: float,
    rho: float
) -> Union[complex, NDArray[complex_]]:
    """
    The function appearing on the right hand side of the Riccati equation for psi_2.

    :param u: characteristic function argument.
    :param psi: psi as a number or array.
    :param nu: model parameter.
    :param lam: model parameter.
    :param rho: model parameter.
    :return: the value of the function F on the time grid `t`.
    """
    return 0.5 * (u ** 2 - u) + psi * (nu * rho * u - lam) + 0.5 * nu ** 2 * psi ** 2


@jit(parallel=True, nopython=True)
def _jit_parallel_char_func(
    t_grid: NDArray[float_],
    u_arr: NDArray[complex_],
    nu: float,
    lam: float,
    rho: float,
    theta: float,
    V0: float
) -> NDArray[complex128]:
    cf_arr = np.zeros(len(u_arr), dtype=complex128)
    for i in prange(len(u_arr)):
        cf_arr[i] = _jit_char_func(t_grid=t_grid, u=u_arr[i], nu=nu, lam=lam, rho=rho, theta=theta, V0=V0)
    return cf_arr


@jit(nopython=True)
def _jit_char_func(
    t_grid: NDArray[float_],
    u: complex,
    nu: float,
    lam: float,
    rho: float,
    theta: float,
    V0: float
) -> complex:
    """
    Calculates the characteristic function in the Heston model.

    psi is a solution to the Riccati equation

    ψ'(t) = F(u, ψ(t)),
    ψ(0) = 0,

    solved with the Euler's scheme. Here, the function F is self._riccati_func().

    :param t_grid: time grid.
    :param u: characteristic function argument.
    :param nu: model parameter.
    :param lam: model parameter.
    :param rho: model parameter.
    :return: psi on `t_grid` as an NDArray of the same size.
    """
    psi = np.zeros(t_grid.size, dtype=np.complex128)

    dt = np.diff(t_grid)
    if np.any(dt < 0):
        raise ValueError("Time grid should be increasing.")

    for i in range(len(t_grid) - 1):
        F_i = _jit_heston_riccati_func(u=u, psi=psi[i], nu=nu, lam=lam, rho=rho)
        psi[i + 1] = psi[i] + dt[i] * F_i

    phi = np.trapz(psi * theta * lam, x=t_grid)

    return np.exp(phi + psi[-1] * V0)
