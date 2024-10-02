from abc import abstractmethod
import numpy as np
from numpy.typing import NDArray
from numpy import float_
from typing import Union, Tuple

from models.model import Model


class MonteCarloModel(Model):
    """
    A generic class that allows for price trajectory simulation.
    """
    @abstractmethod
    def get_price_trajectory(
            self,
            t_grid: NDArray[float_],
            size: int,
            F0: Union[float, NDArray[float_]],
            rng: np.random.Generator = None,
            *args,
            **kwargs
    ) -> Union[NDArray[float_], Tuple[NDArray[float_], ...]]:
        """
        Generic method to simulate the underlying price trajectories on the given time grid.

        :param t_grid: time grid.
        :param size: number of trajectories to simulate.
        :param F0: initial value of the underlying price.
        :param rng: random number generator to simulate the trajectories with.
        :return: an array `F_traj` of shape (size, len(t_grid)) of simulated price trajectories
        """
        raise NotImplementedError()
