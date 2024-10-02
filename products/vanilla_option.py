import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from numpy import float_, complex_
from scipy.stats import norm
from typing import Union, Tuple, List
from dataclasses import dataclass, asdict
from scipy.special import roots_laguerre
from math import ceil
from tqdm import tqdm

from simulation.monte_carlo import MonteCarlo
from models.model import Model
from models.analytic_model import AnalyticModel
from models.characteristic_function_model import CharacteristicFunctionModel
from models.monte_carlo_model import MonteCarloModel
from models.model_params import ModelParams
from utility.utility import is_put, is_call, to_numpy, DEFAULT_SEED
from volatility_surface.volatility_surface import black_iv, black_vanilla_price
from products.product import Product


@dataclass
class VanillaOption(Product):
    """
    A class describing European vanilla options, either calls or puts.

    T: option maturities, a number or a 1-dimensional array.

    K: options strikes. Either a number, or a 1D array of strikes, or a 2D array of shape
        (len(T), len(strikes)) containing in the i-th raw the strikes corresponding to maturity T[i].

    flag: determines the option type: "c" or "call" for calls, "p" or "put" for puts.

    underlying_name: name of the underlying asset that may be used to determine the pricing model.
    """
    T: Union[float, NDArray[float_]]
    K: Union[float, NDArray[float_]]
    flag: str = "call"
    underlying_name: str = None

    def __post_init__(self):
        self.T = np.reshape(self.T, (-1))
        if self.T.shape != (self.T.size,):
            raise ValueError("`T` should be a float or a one-dimensional array.")
        self.K = np.array(self.K)
        if len(self.K.shape) < 2:
            self.K = np.reshape(self.K, (1, -1)) * np.reshape(np.ones_like(self.T), (-1, 1))

    def get_price(
        self,
        model: Model,
        method: str,
        F0: Union[float, NDArray[float_]],
        is_vol_surface: bool = False,
        pricing_params: ModelParams = ModelParams(),
        **kwargs
    ) -> Union[float, NDArray[float_], Tuple[NDArray[float_], ...]]:
        """
        A generic method that calculates the vanilla option price in the given model with the given method.

        :param model: a model used for pricing. If `method` == "lewis", should be
            inherited from `CharacteristicFunctionModel`, if `method` == "mc", should be
            inherited from `MonteCarloModel`, if `method` == "analytic", should be
            inherited from `AnalyticModel`.
        :param method: a method used to price the product. Possible values:
            "lewis" for Lewis method, "mc" for Monte Carlo, "analytic" for explicit formulae if available.
        :param F0: initial value of the underlying price.
        :param is_vol_surface: whether to return the Black implied volatility value instead of option prices.
        :param pricing_params: an instance of pricing params dataclass corresponding to the chosen pricing method
            to be passed to the pricing function.
        :param **kwargs: you must specify T_grid which is used to integrate over the delivery period
        :return: price(s) of the vanilla option of shape consistent with the shape of `T`and `K`.
        """
        if method == 'mc':
            if isinstance(model, MonteCarloModel):
                return self._get_price_mc(model=model, F0=F0, is_vol_surface=is_vol_surface,
                                          **asdict(pricing_params), **kwargs)
            else:
                raise TypeError("A model should have an implemented trajectory simulation function. "
                                "Provide a model inherited from `MonteCarloModel`")
        elif method == 'lewis':
            if isinstance(model, CharacteristicFunctionModel):
                return to_numpy(self._get_price_lewis(
                    model=model, T=self.T, K=self.K, F0=F0, flag=self.flag,
                    is_vol_surface=is_vol_surface, **asdict(pricing_params),
                    **kwargs
                ))
            else:
                raise TypeError("A model should have an implemented characteristic function. "
                                "Provide a model inherited from `CharacteristicFunctionModel`")
        elif method == 'analytic':
            if isinstance(model, AnalyticModel):
                return model.get_vanilla_option_price_analytic(T=self.T, K=self.K, F0=F0, flag=self.flag,
                                                               is_vol_surface=is_vol_surface)
            else:
                raise TypeError("A model should have an implemented analytic pricing formulae. "
                                "Provide a model inherited from `CharacteristicFunctionModel`")
        else:
            raise ValueError("Not valid pricing method.")

    def _get_price_mc(
        self,
        model: MonteCarloModel,
        F0: Union[float, NDArray[float_]],
        is_vol_surface: bool = False,
        size: int = 10**5,
        return_accuracy: bool = False,
        confidence_level: float = 0.95,
        timestep: float = 1 / 250,
        batch_size: int = 10**5,
        rng: np.random.Generator = None,
        **kwargs
    ) -> Union[float, NDArray[float_], Tuple[NDArray[float_], ...]]:
        """
        Calculates the price of the vanilla European option (F_T - K)^+ or (K - F_T)^+ with Monte Carlo simulation.

        :param model:
        :param F0: initial value of the underlying price.
        :param is_vol_surface: whether to return the Black implied volatility value instead of option prices.
        :param size: number of trajectories to simulate.
        :param return_accuracy: whether to return the confidence interval for the prices / vols.
        :param confidence_level: Monte Carlo simulation confidence level.
        :param timestep: time step for the Euler's discretization.
        :param batch_size: batch_size to be used in Monte Carlo simulation.
        :param rng: random number generator to simulate the trajectories with.
        :param dT: integration step for the third dimension of sigmas.
        :return: n array of shape (T.size, K.shape[-1]) with the option prices or implied vols.
        """
        #  merge linspace with given strikes grid and sort it
        t_grid_linspace = np.linspace(0, self.T[-1], ceil(self.T[-1] / timestep))
        t_grid = np.concatenate((t_grid_linspace, self.T))
        sorted_idx = np.argsort(t_grid)
        t_grid = t_grid[sorted_idx]

        #  get the indices corresponding to the elements of T
        sorted_idx_inverse = np.zeros_like(sorted_idx)
        sorted_idx_inverse[sorted_idx] = np.arange(sorted_idx.size)
        maturities_idx = sorted_idx_inverse[len(t_grid_linspace):]

        if is_call(self.flag):
            def payoff(F: NDArray[float_], K: NDArray[float_]):
                return np.maximum(0, F[:, :, None] - K[None, :, :])
        elif is_put(self.flag):
            def payoff(F: NDArray[float_], K: NDArray[float_]):
                return np.maximum(0, K[None, :, :] - F[:, :, None])
        else:
            raise ValueError("Wrong `flag` value was given.")

        batch_size = min(size, batch_size)
        n_batch = ceil(size / batch_size)
        mc = None
        if rng is None:
            rng = np.random.default_rng(seed=DEFAULT_SEED)
        for i in tqdm(range(n_batch)):
            F_T = model.get_price_trajectory(t_grid=t_grid, size=batch_size, F0=F0,
                                             rng=rng, **kwargs)[:, maturities_idx]
            F_T = np.reshape(F_T, (batch_size, self.T.size))
            if i == 0:
                mc = MonteCarlo(batch=payoff(F_T, self.K), confidence_level=confidence_level)
            else:
                mc.add_batch(batch=payoff(F_T, self.K))

        prices = mc.mean
        price_accuracy = mc.accuracy
        lower_bound, upper_bound = prices - price_accuracy, prices + price_accuracy

        if is_vol_surface:
            prices = black_iv(option_price=prices, T=self.T, K=self.K, F=F0, r=0, flag=self.flag)
            lower_bound = black_iv(option_price=lower_bound, T=self.T, K=self.K, F=F0, r=0, flag=self.flag)
            upper_bound = black_iv(option_price=upper_bound, T=self.T, K=self.K, F=F0, r=0, flag=self.flag)

        if return_accuracy:
            return prices.squeeze(), lower_bound.squeeze(), upper_bound.squeeze()
        else:
            return prices.squeeze()

    def _get_price_lewis(
            self,
            model: CharacteristicFunctionModel,
            T: Union[float, NDArray[float_]],
            K: Union[float, NDArray[float_]],
            F0: float,
            is_vol_surface: bool = False,
            N_points: int = 30,
            control_variate_sigma: float = 0.4,
            **kwargs
    ):
        flag = self.flag
        T = np.reshape(T, (-1))
        if T.shape != (T.size,):
            raise ValueError("`T` should be a float or a one-dimensional array.")
        K = to_numpy(K)
        prices = np.zeros((T.size, K.shape[-1]))

        def black_cf(u: NDArray[complex_], T: float):
            return np.exp(-0.5 * control_variate_sigma ** 2 * (u ** 2 + 1j * u) * T)

        for i, maturity in enumerate(T):
            strikes = K[i] if len(K.shape) == 2 else K
            k = np.log(F0 / strikes)
            z_arr, w_arr = roots_laguerre(n=N_points)

            z_arr = np.reshape(z_arr, (-1, 1))
            integrand_arr = (np.exp(1j * (z_arr - 1j / 2) * k.reshape((1, -1))) * (
                    model.characteristic_function(T=maturity, x=0, u1=z_arr - 1j / 2, **kwargs).reshape((-1, 1)) -
                    black_cf(u=z_arr - 1j / 2, T=maturity)
            ) / (z_arr ** 2 + 0.25)).real
            integral = (w_arr * np.exp(z_arr.squeeze())) @ integrand_arr

            prices[i] = black_vanilla_price(sigma=control_variate_sigma, T=maturity, K=strikes, F=F0, r=0, flag='c') - \
                        strikes / np.pi * integral
            if is_put(flag):
                prices[i] += strikes - F0
            if is_vol_surface:
                prices[i] = black_iv(option_price=prices[i], T=maturity, K=strikes, F=F0, r=0, flag=flag)
        return prices.squeeze()

    def black_iv(
            self,
            option_price: Union[float, NDArray[float_]],
            F: Union[float, NDArray[float_]],
            r: Union[float, NDArray[float_]] = 0
    ) -> Union[float, NDArray[float_]]:
        """
        Calculates implied vol in the Black-76 model given the option price and parameters.

        Args:
            :param option_price: option prices.
            :param F: forward prices at t = 0.
            :param r: the risk-free interest rate.

        Returns:
            Implied volatility or an array of implied volatilities corresponding to the prices.
        """
        return black_iv(option_price=option_price, T=self.T, K=self.K, F=F, r=r, flag=self.flag)

    def black_vanilla_price(
            self,
            sigma: Union[float, NDArray[float_]],
            F: Union[float, NDArray[float_]],
            r: Union[float, NDArray[float_]] = 0,
            flag: str = 'c',
    ) -> Union[float, NDArray[float_]]:
        """
        Calculates prices in the Black-76 model given the option vol and parameters.

        Args:
            sigma: black-76 volatility.
            F: forward prices at t = 0.
            r: the risk-free interest rate.
            flag: 'c' for calls, 'p' for puts.

        Returns:
            Option price or an array of option prices corresponding to the given parameters.
        """
        return black_vanilla_price(sigma=sigma, T=np.reshape(self.T, (-1, 1)), K=self.K, F=F, r=r, flag=self.flag)

    def vega(
        self,
        sigma: Union[float, NDArray[float_]],
        F: Union[float, NDArray[float_]],
    ):
        """
        Calculates vega in the Black-76 model.

        :param sigma: black-76 volatility.
        :param F: forward prices at t = 0.
        :return:
            Options vega or an array of options vega corresponding to the given vols.
        """
        T = np.reshape(self.T, (-1, 1))
        return (F * norm.pdf((np.log(F / self.K) + (0.5 * sigma ** 2) * T) /
                (sigma * np.sqrt(T))) * np.sqrt(T)).squeeze()

    def plot_smiles(
        self,
        option_prices_model: NDArray[float_],
        option_prices_market: NDArray[float_] = None,
        F0: float = 1,
        option_prices_low_model: NDArray[float_] = None,
        option_prices_up_model: NDArray[float_] = None,
        ax: plt.Axes = None,
        option_name: str = None,
        bid_ask_spread: Union[float, List[float]] = None
    ) -> None:
        """
        Plot the implied volatility smiles.

        :param option_prices_model: option prices given by the pricing model.
        :param option_prices_market: market option prices if available.
        :param F0: underlying price at t = 0.
        :param option_prices_low_model: lower prices given by the Monte Carlo estimator.
        :param option_prices_up_model: upper prices given by the Monte Carlo estimator.
        :param ax: matplotlib axis to plot on.
        :param option_name: option name to be traced in the title.
        :param bid_ask_spread: bid-ask spread for IV.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))

        option_prices_model = np.reshape(option_prices_model, self.K.shape)
        iv_model = self.black_iv(option_price=option_prices_model, F=F0)

        if option_prices_low_model is not None:
            iv_model_low = self.black_iv(option_price=option_prices_low_model, F=F0)
            iv_model_up = self.black_iv(option_price=option_prices_up_model, F=F0)

        if option_prices_market is not None:
            option_prices_market = np.reshape(option_prices_market, self.K.shape)
            iv_market = self.black_iv(option_price=option_prices_market, F=F0)
            iv_market = np.reshape(iv_market, self.K.shape)

        if isinstance(bid_ask_spread, float):
            bid_ask_spread = [bid_ask_spread] * len(self.T)

        for i, (maturity, strikes) in enumerate(zip(self.T, self.K)):
            log_mon = np.log(strikes / F0)
            title_option = f'option={option_name}' if option_name is not None else ""
            if option_prices_market is not None:
                if option_prices_market is not None:
                    ax.plot(log_mon, iv_market[i], 'v--', color="r", label="market")
                if bid_ask_spread is not None:
                    ax.fill_between(log_mon, iv_market[i] - 0.5 * np.array(bid_ask_spread[i]),
                                    iv_market[i] + 0.5 * np.array(bid_ask_spread[i]), color="r",
                                    label="bid-ask", alpha=0.2)
            ax.plot(log_mon, iv_model[i], 'v--', ms=4, color="b", label="model")
            if option_prices_low_model is not None:
                ax.fill_between(log_mon, iv_model_low[i], iv_model_up[i], color="b",
                                label="MC confidence intervals", alpha=0.2)

            ax.grid('on')
            ax.legend()
            ax.set_xlabel(r'$\log(K / F0)$')
            ax.set_title(title_option + f' and TTM={np.round(maturity, 2)}')

