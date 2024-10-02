import numpy as np
from typing import Union
from numpy.typing import NDArray
from numpy import float_
from utility.utility import DEFAULT_SEED


def simulate_brownian_motion_from_increments(
    size: int,
    t_grid: Union[float, NDArray[float_]],
    dim: int = 1,
    rng: np.random.Generator = None
) -> Union[float, NDArray[float_]]:
    """
    Simulates the trajectory of the standard d-dimensional Brownian motion with the increments method.

    Args:
        t_grid: time grid to simulate the price on.
        size: number of simulated trajectories.
        dim: dimensionality of the Brownian motion.
        rng: `np.random.Generator` used for simulation.

    Returns:
        np.ndarray of shape (size, len(t_grid)) with simulated trajectories if model dimension is 1.
        np.ndarray of shape (size, dim, len(t_grid)) with simulated trajectories if model dimension greater than 1.
    """
    if rng is None:
        rng = np.random.default_rng(seed=DEFAULT_SEED)

    dt = np.diff(np.concatenate([np.zeros(1), t_grid]))
    if np.any(dt < 0):
        raise ValueError("Time grid should be increasing.")
    dW = rng.normal(size=(size, dim, len(t_grid)))
    return np.cumsum(np.sqrt(dt) * dW, axis=2)


# TODO: Add simulation as a brownian bridge
