import numpy as np
from typing import Union, Callable
from numpy.typing import NDArray
from numpy import float_

from utility.utility import to_numpy, DEFAULT_SEED
from simulation.simulation import simulate_brownian_motion_from_increments


class Diffusion:
    t_grid: NDArray[float_]
    W_traj: NDArray[float_]
    rng: np.random.Generator
    dim: int
    size: int

    def __init__(
        self,
        t_grid: NDArray[float_],
        size: int = 1,
        dim: int = 1,
        rng: np.random.Generator = None,
        W_traj: NDArray[float_] = None,
        method: str = 'increments'
    ) -> None:
        """
        Initializes the diffusion object simulating the trajectory of the underlying 'dim'-dimensional
        standard Brownian motion of shape (size, dim, len(t_grid)).

        Args:
            t_grid: time grid to simulate the price on.
            size: number of simulated trajectories.
            dim: dimensionality of the Brownian motion.
            rng: `np.random.Generator` used for simulation.
            W_traj: trajectories of the standard Brownian motion if the simulation is not needed.
            method: method used for Brownian motion simulation. Possible options: 'increments'.
        """
        self.t_grid = to_numpy(t_grid)
        self.rng = np.random.default_rng(seed=DEFAULT_SEED) if rng is None else rng
        if W_traj is not None:
            self.W_traj = W_traj
            self.size = W_traj.shape[0]
            self.dim = W_traj.shape[1]
        else:
            self.size = size
            self.dim = dim
            if method == 'increments':
                self.W_traj = simulate_brownian_motion_from_increments(
                    size=self.size,
                    t_grid=self.t_grid,
                    dim=self.dim,
                    rng=self.rng
                )

    def replace_brownian_motion(
        self,
        W_traj: NDArray[float_],
    ) -> None:
        """
        Reinitializes the diffusion object by a new trajectory of
        standard brownian motion of shape (size, dim, len(t_grid)).

        Args:
            W_traj: trajectories of the standard Brownian motion if the simulation is not needed.
        """
        self.W_traj = W_traj
        self.size = W_traj.shape[0]
        self.dim = W_traj.shape[1]
        if W_traj.shape[2] != len(self.t_grid):
            raise ValueError("The shape of the new Brownian motion should match existing 't_grid' shape.")

    def brownian_motion_increments(
        self,
        dims: Union[float, NDArray[float_]] = None,
        squeeze: bool = False
    ) -> NDArray[float_]:
        """
        Returns the increments of the underlying brownian motion of shape (size, len(dims), len(t_grid)).
        Additional point t = 0 is added to the time grid to calculate the increments.

        Args:
            dims: which dimensions of the underlying standard BM to use for simulation. By default, all.
            squeeze: whether to squeeze the output.
        """
        if dims is None:
            dims = np.arange(self.dim)
        dW = np.diff(
            np.concatenate([np.zeros(self.W_traj[:, dims, :].shape[:2])[:, :, None], self.W_traj[:, dims, :]], axis=2),
            axis=2
        )
        return dW.squeeze() if squeeze else dW

    def brownian_motion(
        self,
        init_val: Union[float, NDArray[float_]] = 0,
        drift: Union[float, NDArray[float_]] = 0,
        correlation: Union[float, NDArray[float_]] = None,
        vol: Union[float, NDArray[float_]] = 1,
        dims: Union[float, NDArray[float_]] = None,
        squeeze: bool = False
    ) -> Union[float, NDArray[float_]]:
        """
        Simulates the trajectory of the d-dimensional shifted correlated Brownian motion.

        Args:
            init_val: value of the process at t = 0.
            drift: number or vector of size d representing the constant drift.
            correlation: correlation matrix of the increments per unit time.
            vol: volatility of the Brownian motion.
            dims: which dimensions of the underlying standard BM to use for simulation. By default all.
            squeeze: whether to squeeze the output.

        Returns:
            np.ndarray of shape (size, len(dims), len(t_grid)) with simulated trajectories.
        """
        drift = to_numpy(drift)
        init_val = to_numpy(init_val)
        vol = to_numpy(vol)
        if dims is None:
            dims = np.arange(self.dim)
        if correlation is None:
            L = np.eye(len(dims))
        else:
            L = np.linalg.cholesky(correlation)
        traj = init_val[None, :, None] + drift[None, :, None] * self.t_grid[None, None, :] + \
            vol[None, :, None] * np.einsum('ij,kjl->kil', L, self.W_traj[:, dims, :])
        return traj.squeeze() if squeeze else traj

    def geometric_brownian_motion(
        self,
        init_val: Union[float, NDArray[float_]] = 1,
        drift: Union[float, NDArray[float_]] = 0,
        correlation: Union[float, NDArray[float_]] = None,
        vol: Union[float, NDArray[float_]] = 1,
        dims: Union[float, NDArray[float_]] = None,
        squeeze: bool = False
    ) -> Union[float, NDArray[float_]]:
        """
        Simulates the trajectory of the d-dimensional geometric Brownian motion.

        Args:
            init_val: value of the process at t = 0.
            drift: number or vector of size d such that E[X_T] = exp(drift * T).
            correlation: correlation matrix of log-increments per unit time.
            vol: volatility of the log-process.
            dims: which dimensions of the underlying standard BM to use for simulation. By default all.
            squeeze: whether to squeeze the output.

        Returns:
            np.ndarray of shape (size, len(dims), len(t_grid)) with simulated trajectories.
        """
        drift = to_numpy(drift)
        init_val = to_numpy(init_val)
        vol = to_numpy(vol)
        drift_log = drift - 0.5 * vol**2
        W = self.brownian_motion(
            init_val=0,
            drift=drift_log,
            correlation=correlation,
            vol=vol,
            dims=dims,
            squeeze=False
        )
        traj = init_val[None, :, None] * np.exp(W)
        return traj.squeeze() if squeeze else traj

    def integral_of_brownian_motion(
        self,
        T: float,
        dims: Union[float, NDArray[float_]] = None,
        squeeze: bool = False
    ) -> NDArray[float_]:
        """
                  T
        Simulates ∫ W_t dt given the trajectory of W_t on 't_grid'.
                  0
        Args:
            T: upper limit of the integral.
            dims: which dimensions of the underlying standard BM to use for simulation. By default all.
            squeeze: whether to squeeze the output.
        """
        if dims is None:
            dims = np.arange(self.dim)
        dW = self.brownian_motion_increments(dims)
        dt = np.diff(np.concatenate([np.zeros(1), self.t_grid]))
        beta = 0.5 * dt + (T - self.t_grid)
        if T < np.max(self.t_grid):
            idx_end = np.where(self.t_grid > T)[0][0]
            dW = dW[:, :, :idx_end + 1]
            dt = dt[:idx_end + 1]
            beta = beta[:idx_end + 1]
            beta[idx_end] = (T - self.t_grid[idx_end])**2 / (2 * dt[idx_end])
        m = np.einsum('i,kji->kj', beta, dW)
        v = T**3 / 3 - beta**2 @ dt
        integral = np.sqrt(v) * self.rng.normal(size=dW.shape[:2]) + m
        return integral.squeeze() if squeeze else integral
