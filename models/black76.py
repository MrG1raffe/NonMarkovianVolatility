import numpy as np
from dataclasses import dataclass
from scipy.stats import norm
from typing import Union, Tuple
from numpy.typing import NDArray
from numpy import float_, complex_

from simulation.diffusion import Diffusion
from models.monte_carlo_model import MonteCarloModel
from models.analytic_model import AnalyticModel
from models.characteristic_function_model import CharacteristicFunctionModel
from utility.utility import to_numpy, is_put, is_call
from volatility_surface.volatility_surface import black_iv


@dataclass
class Black76(AnalyticModel, CharacteristicFunctionModel, MonteCarloModel):
    sigma: float

    def _d1d2(
        self,
        T: Union[float, NDArray[float_]],
        K: Union[float, NDArray[float_]],
        F: Union[float, NDArray[float_]]
    ) -> Tuple[Union[float, NDArray[float_]], Union[float, NDArray[float_]]]:
        """
        Calculates d1 and d2 from Black-76 formula.

        Args:
            T: times to maturity.
            K: strikes.
            F: forward prices at t = 0.

        Returns:
            d1: values of d1.
            d2: values of d2.
        """
        d1 = (np.log(F/K) + (self.sigma**2/2)*T) / (self.sigma*np.sqrt(T))
        d2 = d1 - self.sigma * np.sqrt(T)
        return d1, d2

    def get_vanilla_option_price_analytic(
        self,
        T: Union[float, NDArray[float_]],
        K: Union[float, NDArray[float_]],
        F0: float,
        flag: str = "call",
        is_vol_surface: bool = False
    ) -> Union[float, NDArray[float_]]:
        """
        Calculates the vanilla option price via Black-76 formula

        Args:
            T: times to maturity.
            K: strikes.
            F0: forward prices at t = 0.
            flag: 'c' for calls, 'p' for puts.
            is_vol_surface: whether to return the Black implied volatility value instead of option prices.

        Returns:
            Prices of the call/put vanilla options.
        """
        T = np.reshape(T, (-1, 1))
        K = to_numpy(K)
        if len(K.shape) < 2:
            K = np.reshape(K, (1, -1))
        d1, d2 = self._d1d2(T, K, F0)
        if is_call(flag):
            prices = (F0 * norm.cdf(d1) - K * norm.cdf(d2))
        elif is_put(flag):
            prices = (-F0 * norm.cdf(-d1) + K * norm.cdf(-d2))
        else:
            raise ValueError("Incorrect flag value was given.")
        if is_vol_surface:
            prices = black_iv(option_price=prices, T=T * np.ones_like(K), K=np.ones_like(T) * K,
                              F=F0, r=0, flag=flag)
        return prices.squeeze()

    def delta(
        self,
        T: Union[float, NDArray[float_]],
        K: Union[float, NDArray[float_]],
        F: Union[float, NDArray[float_]],
        flag: str
    ) -> Union[float, NDArray[float_]]:
        """
        Calculates the option delta in the Black-76 model

        Args:
            T: times to maturity.
            K: strikes.
            F: forward prices at t = 0.
            flag: 'c' for calls, 'p' for puts.

        Returns:
            Vega of the option(s).
        """
        d1, _ = self._d1d2(T, K, F)
        return norm.cdf(d1) if flag == 'c' else -norm.cdf(-d1)

    def gamma(
        self,
        T: Union[float, NDArray[float_]],
        K: Union[float, NDArray[float_]],
        F: Union[float, NDArray[float_]],
    ) -> Union[float, NDArray[float_]]:
        """
        Calculates the option gamma in the Black-76 model

        Args:
            T: times to maturity.
            K: strikes.
            F: forward prices at t = 0.

        Returns:
            Vega of the option(s).
        """
        d1, _ = self._d1d2(T, K, F)
        return norm.pdf(d1) / (F * self.sigma * np.sqrt(T))

    def vega(
        self,
        T: Union[float, NDArray[float_]],
        K: Union[float, NDArray[float_]],
        F: Union[float, NDArray[float_]],
    ) -> Union[float, NDArray[float_]]:
        """
        Calculates the option vega in the Black-Scholes model

        Args:
            T: times to maturity.
            K: strikes.
            F: spot prices at t = 0.

        Returns:
            Vega of the option(s).
        """
        d1, _ = self._d1d2(T, K, F)
        return F * norm.pdf(d1) * np.sqrt(T)

    def get_price_trajectory(
        self,
        t_grid: NDArray[float_],
        size: int,
        F0: Union[float, NDArray[float_]],
        rng: np.random.Generator = None,
        *args,
        **kwargs
    ) -> Union[float, NDArray[float_]]:
        """
        Simulates the trajectory of stock or forward in the Black model.

        Args:
            size: number of simulated trajectories.
            t_grid: time grid to simulate the price on.
            F0: the underlying price at t = 0.
            rng: `np.random.Generator` used for simulation.

        Returns:
            np.ndarray of shape (size, len(t_grid)) with simulated trajectories if model dimension is 1.
            np.ndarray of shape (size, dim, len(t_grid)) with simulated trajectories if model dimension greater than 1.
        """
        diffusion = Diffusion(
            t_grid=t_grid,
            size=size,
            dim=1,
            rng=rng
        )
        F_traj = diffusion.geometric_brownian_motion(
            init_val=F0,
            vol=self.sigma,
            squeeze=True
        )
        return F_traj

    def characteristic_function(
        self,
        T: float,
        x: float,
        u1: Union[complex, NDArray[complex_]],
        u2: Union[complex, NDArray[complex_]] = 0,
        f1: Union[complex, NDArray[complex_]] = 0,
        f2: Union[complex, NDArray[complex_]] = 0,
        **kwargs
    ) -> Union[complex, NDArray[complex_]]:
        """
        Computes the generalized characteristic function

        E[exp{i * u1 * log(F_T) + i * u2 * V_T + i * f1 * ∫ log(F_s) ds + i * f2 * ∫ V_s ds}]     (1)

        for the Bachelier model, where V_t = σ.

        :param u1: F_T coefficient in the characteristic function, see (1).
        :param u2: V_T coefficient in the characteristic function, see (1).
        :param f1: ∫ F_s ds coefficient in the characteristic function, see (1).
        :param f2: ∫ V_s ds coefficient in the characteristic function, see (1).
        :param T: date in the characteristic function, see (1).
        :param x: X_0, equals to log(F_0).
        :return: a value of the characteristic function (1) for the given coefficients.
        """
        # TODO: verify the joint CF for Asian options.
        return np.exp(1j * u1 * x - 0.5 * self.sigma**2 * T * (u1**2 + (f1 * T)**2 / 3 + u1 * f1 * T) +
                      1j * u2 + 1j * f2 * T) * np.exp(-0.5 * self.sigma**2 * T * (1j * u1 + 0.5 * T * 1j * f1))
