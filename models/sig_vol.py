import numpy as np
from numpy.typing import NDArray
from numpy import float64, complex128
from numba import jit, njit, prange
from dataclasses import dataclass
from typing import Union, Tuple, Callable, Iterable
from math import ceil
from tqdm import tqdm

from signatures.tensor_sequence import TensorSequence
from signatures.tensor_algebra import TensorAlgebra, Alphabet
from models.characteristic_function_model import CharacteristicFunctionModel
from simulation.diffusion import Diffusion
from utility.utility import DEFAULT_SEED


@dataclass
class SigVol(CharacteristicFunctionModel):
    vol_ts: TensorSequence
    ta: TensorAlgebra
    sigmas: Callable[[NDArray[float64]], NDArray[float64]]
    R: NDArray[float64]
    rhos: NDArray[float64]
    model_type: str

    def __post_init__(self):
        if self.model_type not in ["log-normal"]:
            raise ValueError("`model_type` should be 'log-normal'.")

        # convert everything to numpy
        self.rhos = np.array(self.rhos)
        self.R = np.array(self.R)

        # check the dimensions
        self.n_hist_factors = self.rhos.size

        if self.R.shape != (self.n_hist_factors, self.n_hist_factors) or \
                not (self.sigmas(np.zeros(1)).size == self.n_hist_factors or
                     self.sigmas(np.zeros(1)).shape[1] == self.n_hist_factors):
            raise ValueError("Inconsistent historical factors dimensions.")

        # transform the correlation with independent Brownian motions to the correlation
        # with the factors W correlated with matrix R
        if np.sum(self.rhos ** 2) > 1:
            raise ValueError("Inadmissible value of `rhos` was given. It should satisfy ||rhos|| <= 1.")
        L = np.linalg.cholesky(self.R)
        self.L = L
        self.rhos_WB = L @ self.rhos

    def _char_func(
        self,
        T: float,
        x: float,
        u1: complex,
        u2: complex = 0,
        f1: complex = 0,
        f2: complex = 0,
        cf_timestep: float = 0.001,
        max_grid_size: int = 10 ** 8,
        scheme: str = "Euler",
        **kwargs
    ) -> Union[complex128, NDArray[complex128]]:
        """
        Computes the generalized characteristic function

        E[exp{i * u1 * X_T}]     (1)

        for the given model, where X_t = F_t if `model_type` == "normal" and
        X_t = log(F_t) if `model_type` == "log-normal".

        :param u1: X_T coefficient in the characteristic function, see (1).
        :param T: date in the characteristic function, see (1).
        :param x: X_0, equals to F_0 if `model_type` == "normal" and to log(F_0) if `model_type` == "log-normal".
        :return: a value of the characteristic function (1) for the given coefficients.
        """
        if (not np.isclose(u2, 0) or not np.isclose(f1, 0) or
                not np.isclose(f2, 0) or self.model_type != "log-normal"):
            raise NotImplementedError()

        u1 = 1j * u1
        timestep = max(min(cf_timestep, T / 10), T / max_grid_size)

        t_grid = np.linspace(0, T, ceil(T / timestep) + 1)

        sigmas_arr = self.sigmas(t_grid[-1] - t_grid)
        spot_vol_cov = sigmas_arr @ self.rhos_WB
        det_var = np.sum(sigmas_arr.T * (self.R @ sigmas_arr.T), 0)

        if isinstance(u1, Iterable):
            return jit_parallel_char_func(t_grid=t_grid, u_arr=np.array(u1), vol_ts=self.vol_ts,
                                          trunc=2 * self.vol_ts.trunc, spot_vol_cov=spot_vol_cov, det_var=det_var)
        else:
            return jit_char_func(t_grid=t_grid, u=u1, vol_ts=self.vol_ts, trunc=2 * self.vol_ts.trunc,
                                 spot_vol_cov=spot_vol_cov, det_var=det_var)

    def quadratic_variation(
        self,
        T: float,
    ) -> float:
        """
        Computes the quadratic variation of the process <X>_T at t = T.

        :param T: the date the quadratic variation to be calculated on.
        :return: the value of numerical approximation of <X>_T.
        """
        # TODO: implement expected signature to calculate E[V_t]
        raise NotImplementedError()

    def get_corr_mat(self) -> NDArray[float64]:
        """
        :return: a complete correlation matrix of the Brownian motion (W_t, B_t).
        """
        corr_mat = np.eye(self.n_hist_factors + 1)
        corr_mat[:self.n_hist_factors, :self.n_hist_factors] = self.R
        corr_mat[-1, :-1] = self.rhos_WB
        corr_mat[:-1, -1] = self.rhos_WB
        return corr_mat

    def get_vol_trajectory(
        self,
        t_grid: NDArray[float64],
        size: int,
        rng: np.random.Generator = None,
        B_traj: NDArray[float64] = None
    ) -> NDArray[float64]:
        """
        Simulate the variance and the factor processes on the given time grid.

        :param t_grid: time gird.
        :param size: number of trajectories to simulate.
        :param rng: random number generator to simulate the trajectories with.
        :param B_traj: pre-simulated trajectories of the BM B_t corresponding to the stochastic volatility.
            By default, None, and will be simulated within the function.
        :return: an array of shape (size, len(t_grid)) with the volatility trajectories.
        """
        if B_traj is None:
            # simulation of B_traj
            diffusion = Diffusion(t_grid=t_grid, dim=1, size=size, rng=rng)
            B_traj = diffusion.brownian_motion()[:, 0, :]  # shape (size, len(t_grid))
        else:
            if B_traj.shape != (size, len(t_grid)):
                raise ValueError("Inconsistent dimensions of B_traj were given.")

        path = np.zeros((t_grid.size, 2, size))
        path[:, 0, :] = np.reshape(t_grid, (-1, 1))
        path[:, 1, :] = B_traj.T
        return np.real(self.ta.path_to_sequence(path=path, trunc=self.vol_ts.trunc) @ self.vol_ts).T

    def get_price_trajectory(
        self,
        t_grid: NDArray[float64],
        size: int,
        F0: Union[float, NDArray[float64]],
        rng: np.random.Generator = None,
        return_vol: bool = False,
        **kwargs
    ) -> Union[NDArray[float64], Tuple[NDArray[float64], ...]]:
        """
        Simulates the underlying price trajectories on the given time grid.

        :param t_grid: time grid.
        :param size: number of trajectories to simulate.
        :param F0: initial value of the underlying price.
        :param rng: random number generator to simulate the trajectories with.
        :param return_vol: whether to return the volatility trajectory together with the prices.
        :return: an array `F_traj` of shape (size, len(t_grid)) of simulated price trajectories,
            an array `sigma_traj` of shape (size, len(t_grid)) of volatility trajectories if `return_vol` == True.
        """

        if rng is None:
            rng = np.random.default_rng(seed=DEFAULT_SEED)

        diffusion = Diffusion(t_grid=t_grid, dim=self.n_hist_factors + 1, size=size, rng=rng)

        corr_mat = self.get_corr_mat()

        brownian_motion = diffusion.brownian_motion(correlation=corr_mat)

        dW_traj = np.diff(brownian_motion[:, :self.n_hist_factors, :], axis=2)  # shape (size, n_hist_factors, len(t_grid)-1)
        B_traj = brownian_motion[:, self.n_hist_factors, :]  # shape (size, len(t_grid))

        vol_traj = self.get_vol_trajectory(t_grid=t_grid, size=size, B_traj=B_traj)

        sigmas = self.sigmas(t_grid[:-1])  # shape (len(t_grid)-1, n_hist_factors)

        log_F_traj = np.log(F0) * np.ones((size, len(t_grid)))
        log_F_traj[:, 1:] += np.cumsum(
            np.einsum('ki,jik->jk', sigmas, dW_traj) * vol_traj[:, :-1],
            axis=1
        )
        log_F_traj[:, 1:] -= 0.5 * np.cumsum(
            np.sum((t_grid[1:] - t_grid[:-1]) * sigmas.T * (self.R @ sigmas.T), axis=0) * vol_traj[:, :-1]**2,
            axis=1
        )
        F_traj = np.exp(log_F_traj)

        if return_vol:
            return F_traj, vol_traj
        return F_traj


@njit(parallel=True)
def jit_parallel_char_func(
    t_grid: NDArray[float64],
    u_arr: NDArray[complex128],
    vol_ts: TensorSequence,
    trunc: int,
    spot_vol_cov: NDArray[float64],
    det_var: NDArray[float64]
) -> NDArray[complex128]:
    cf_arr = np.zeros(len(u_arr), dtype=complex128)
    for i in prange(len(u_arr)):
        cf_arr[i] = np.reshape(jit_char_func(t_grid=t_grid, u=u_arr[i], vol_ts=vol_ts, trunc=trunc,
                                             spot_vol_cov=spot_vol_cov, det_var=det_var), ())
    return cf_arr


@jit(nopython=True)
def jit_char_func(
    t_grid: NDArray[float64],
    u: complex,
    vol_ts: TensorSequence,
    trunc: int,
    spot_vol_cov: NDArray[float64],
    det_var: NDArray[float64]
) -> complex128:
    dt = np.diff(t_grid)
    vol_shuffle_squared = vol_ts.shuffle_pow(2) * (0.5 * (u**2 - u))

    psi = TensorSequence(Alphabet(2), trunc, np.zeros(1), np.zeros(1))
    for i in range(len(dt)):
        psi = psi + (
            psi.proj("2").shuffle_pow(2) / 2 + vol_ts.shuffle_prod(psi.proj("2")) * (u * spot_vol_cov[i]) +
            psi.proj("22") / 2 + psi.proj("1") + vol_shuffle_squared * det_var[i]
        ) * dt[i]

    return np.exp(psi[""])
