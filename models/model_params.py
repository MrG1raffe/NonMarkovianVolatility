import numpy as np
from dataclasses import dataclass
from numpy.typing import NDArray
from numpy import float_


@dataclass
class ModelParams:
    """
    A generic class for stochastic volatility model parameters.
    """


@dataclass
class HestonParams(ModelParams):
    """
    Heston model parameters.
    """
    V0: float = 1
    theta: float = 1
    lam: float = 0
    nu: float = 1
    rho: float = 0


@dataclass
class FractionalVolterraHestonParams(HestonParams):
    """
    Heston model parameters.
    """
    H: float = 0.5
    eps: float = 0


@dataclass
class LiftedHestonParams(HestonParams):
    """
    Heston model parameters.
    """
    c: NDArray[float_] = None
    x: NDArray[float_] = None
    H: float = None
    n: int = None
    r: float = None


@dataclass
class PricingParams:
    """
    A generic class for product pricing parameters.
    """


@dataclass
class LewisParams(PricingParams):
    """
    Parameters for the COS-method.

    N_trunc: number of terms in the Cosine series to be calculated.
    cf_timestep: a timestep to be used in numerical scheme in the characteristic function.
    scheme: numerical scheme for the Riccati equation. Either "exp" or "semi-implicit".
    """
    N_points: int = 20
    cf_timestep: float = 0.003
    max_grid_size: int = 10000
    scheme: str = "exp"
    control_variate_sigma: float = 0.4


@dataclass
class MCParams(PricingParams):
    """
    Parameters for the Monte Carlo pricing

    size: number of trajectories to simulate.
    return_accuracy: whether to return the confidence interval for the prices / vols.
    confidence_level: Monte Carlo simulation confidence level.
    timestep: time step for the Euler's discretization.
    batch_size: batch_size to be used in Monte Carlo simulation.
    rng: random number generator to simulate the trajectories with.
    """
    size: int = 10 ** 5
    return_accuracy: bool = False
    confidence_level: float = 0.95
    timestep: float = 1 / 250
    batch_size: int = 10 ** 5
    rng: np.random.Generator = None
    scheme: str = "exp"

