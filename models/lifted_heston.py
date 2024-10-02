import numpy as np
from numpy.typing import NDArray
from numpy import float_, complex_
from typing import Union, Callable, Tuple, List
from numpy.polynomial.polynomial import polyfromroots
from scipy.linalg import circulant
from scipy.interpolate import lagrange
from dataclasses import dataclass
from math import ceil
from numba import jit

from models.characteristic_function_model import CharacteristicFunctionModel
from simulation.diffusion import Diffusion
from utility.utility import DEFAULT_SEED


@dataclass
class LiftedHeston(CharacteristicFunctionModel):
    """
    A class describing the Lifted Heston model

    dF_t(/F_t) = sqrt(V_t / ξ_t) * sigmas(t) * dW_t,
    V_t = V_0 + ∫ Σ c_i * exp{-x_i * (t - s)} * (λ * (θ - V_s) ds + ν * sqrt(V_s) * dB_s).

    theta: see the equation.
    lam: see the equation.
    nu: see the equation.
    rhos: correlation between B_t and independent Brownian motions the linear combination of which forms W_t.
        After the initialization the correlations between B_t and W_t is calculated and kept as `self.rhos_WB`.
    sigmas: a function of t that returns the deterministic volatility array of shape (len(t), n_hist_factors) or
        of shape or (len(t_grid), n_hist_factors, len(T_grid)) if `sigmas` is 3-dimensional.
    R: correlation matrix of W_t.
    x: mean-reversion coefficients in the kernel, see the equation.
    c: coefficients in the kernel, see the equation.
    V0: initial value of variance.
    model_type: whether the underlying price follows normal or log-normal dynamics. Either "normal" or "log-normal".
    normalize_variance: whether to normalize the variance by ξ_t in the price dynamics.
    is_kemna_vorst: whether the model corresponds to the KV approximation (two-dimensional sigmas) or the true
        forward price model (three-dimensional sigmas).
    kemna_vorst_grid: if is_kemna_vorst == True, defines the integration grid in T used to simulate the trajectories.
    """
    theta: Union[float, NDArray[float_]]
    lam: float
    nu: float
    rhos: NDArray[float_]
    sigmas: Callable[[NDArray[float_]], NDArray[float_]]
    R: NDArray[float_]
    x: NDArray[float_]
    c: NDArray[float_]
    V0: float
    model_type: str
    normalize_variance: bool = False
    is_kemna_vorst: bool = True
    kemna_vorst_grid: List = None

    def __post_init__(self):
        if self.model_type not in ["normal", "log-normal"]:
            raise ValueError("`model_type` should be either 'normal' or 'log-normal'.")

        # convert everything to numpy
        self.rhos = np.array(self.rhos)
        self.R = np.array(self.R)
        self.x = np.array(self.x)
        self.c = np.array(self.c)
        
        # check the dimensions
        self.n_hist_factors = self.rhos.size
        self.n_stoch_factors = self.c.size

        if self.R.shape != (self.n_hist_factors, self.n_hist_factors) or \
                not (self.sigmas(np.zeros(1)).size == self.n_hist_factors or
                     self.sigmas(np.zeros(1)).shape[1] == self.n_hist_factors):
            raise ValueError("Inconsistent historical factors dimensions.")
        if self.x.size != self.n_stoch_factors:
            raise ValueError("Inconsistent stochastic factors dimensions.")

        # transform the correlation with independent Brownian motions to the correlation
        # with the factors W correlated with matrix R
        if np.sum(self.rhos**2) > 1:
            raise ValueError("Inadmissible value of `rhos` was given. It should satisfy ||rhos|| <= 1.")
        L = np.linalg.cholesky(self.R)
        self.L = L
        self.rhos_WB = L @ self.rhos

        # precompute the resolvent of K to needed for the forward variance.
        self.__set_resolvent_params()

    def __set_resolvent_params(self) -> None:
        """
        The resolvent of the second kind satisfying
        R + λ K * R + λ K = 0  (star denotes convolution).
        In the case where K is the sum of exponential kernels, one can reduce this convolution equation to the ODE
        of order n_stochastic_factors. Its solution can be found in the form R(t) = Σ α_i * exp(β_i * t).
        The function computes arrays (α_i) and (β_i) and writes it to the attributes `__resolvent_alphas`
        and `__resolvent_betas`.

        :return: None
        """
        if self.n_stoch_factors == 1:
            # explicit formula for L=1 is available
            betas = -(self.lam * self.c + self.x)
            alphas = -self.lam * self.c
        elif self.n_stoch_factors == 2:
            # explicit formula for L=1 is available
            B = np.sum(self.x) + self.lam * self.__k_der_0(0)
            C = np.prod(self.x) + np.sum(self.x) * self.lam * self.__k_der_0(0) + self.lam * self.__k_der_0(1)
            D = B ** 2 - 4 * C

            # roots of the characteristic polynomial for R
            betas = 0.5 * (-B - np.sqrt(D)) * np.ones(2)
            betas[1] += np.sqrt(D)

            b = [-self.lam * self.__k_der_0(0), -self.lam * self.__k_der_0(1) + self.lam ** 2 * self.__k_der_0(0) ** 2]
            alphas = np.array([b[0] * betas[1] - b[1], b[1] - b[0] * betas[0]]) / (betas[1] - betas[0])
        else:

            # in general case, the roots of the characteristic polynomial are found numerically.
            K_der_lam = self.__k_der_0(np.arange(self.n_stoch_factors)) * self.lam
            p = polyfromroots(-self.x)
            P = np.flip(np.triu(circulant(np.flip(p)).T), axis=1)
            # construct the coefficients of the ODE on R
            betas_polynom = P @ np.concatenate([[1], K_der_lam])
            betas = np.roots(np.flip(betas_polynom))

            # define the initial conditions of R^(l) for l = 0, ..., L.
            R_der_0 = np.zeros(len(betas) + 1)
            R_der_0[0] = 1
            L = np.zeros((self.n_stoch_factors, self.n_stoch_factors))
            for i in range(len(betas)):
                R_der_0[i + 1] = -K_der_lam[:i + 1] @ np.flip(R_der_0[:i + 1])
                e_i = np.zeros_like(betas)
                e_i[i] = 1.0
                # rows of the inverse Vandermonde matrix correspond to the coefficients of the Lagrange polynomials
                L[i] = lagrange(betas, e_i).coef[::-1]
            alphas = L @ R_der_0[1:]
        self.__resolvent_betas = betas.squeeze()
        self.__resolvent_alphas = alphas.squeeze()

    def __k_der_0(self, der_order: Union[int, NDArray[float_]]):
        """
        Calculates the kernel derivative at t=0 of order `der_order`.

        :param der_order: derivative order.
        :return: value of K^(der_order)(0).
        """
        return np.sum(self.c * (-self.x) ** np.reshape(der_order, (-1, 1)), axis=1)

    def forward_variance(
            self,
            t_grid: NDArray[float_]
    ) -> NDArray[float_]:
        """
        Calculates the forward variance ξ_t via the resolvent R which parameters
        are computed in `__set_resolvent_params`.

        :param t_grid: array of t.
        :return: the forward variance ξ_t on `t_grid`.
        """
        return _jit_forward_variance(
            V0=self.V0, theta=self.theta,
            alphas=self.__resolvent_alphas, betas=self.__resolvent_betas,
            t_grid=t_grid)

    def deterministic_variance(
        self,
        t_grid: NDArray[float_]
    ):
        """
        Calculates the deterministic variance sigma.T @ R @ sigma on the given time grid.

        :param t_grid: array defining the time grid.
        :return: array with the deterministic variance on `t_grid`.
        """
        sigmas = self.sigmas(t_grid)
        return np.sum(sigmas.T * (self.R @ sigmas.T), axis=0)

    def quadratic_variation(
            self,
            T: float,
            T0: float = 0,
            dt: float = 0.001
    ) -> float:
        """
        Computes the quadratic variation of the process <X>_T between T0 and T.

        :param T0: start date.
        :param T: the date the quadratic variation to be calculated on.
        :param dt: time step to be used in numerical integration of the volatility.
        :return: the value of numerical approximation of <X>_T.
        """
        t_grid = np.linspace(T0, T, ceil((T - T0) / dt))
        sigmas = self.sigmas(t_grid)
        xi = 1 if self.normalize_variance else self.forward_variance(t_grid)

        return _jit_quadratic_variation(sigmas=sigmas, R=self.R, xi=xi, t_grid=t_grid)

    def vs_vol(
            self,
            T: float,
            dt: float = 0.001
    ) -> float:
        """
        Computes the variance swap volatility of the price process at T.

        :param T: the date the VS vol to be calculated on.
        :param dt: time step to be used in numerical integration of the volatility.
        :return: the value of numerical approximation of the VS volatility.
        """
        return np.sqrt(self.quadratic_variation(T=T, dt=dt) / T)

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

    def spot_vol_correlation(
        self,
        t_grid: NDArray[float_]
    ) -> NDArray[float_]:
        """
        Calculates instantaneous correlation between F_t and V_t on the given time grid.

        :param t_grid: time grid
        :return: array of instantaneous correlations.
        """
        L = np.linalg.cholesky(self.R)
        rho_inst = self.rhos @ ((L.T @ self.sigmas(t_grid).T) / np.linalg.norm((L.T @ self.sigmas(t_grid).T), axis=0))
        return rho_inst

    def _riccati_func(
            self,
            t: NDArray[float_],
            psi_1: NDArray[complex_],
            psi_2: NDArray[complex_],
            f2: complex,
            T: float
    ) -> NDArray[complex_]:
        """
        The function appearing on the right hand side of the Riccati equation for psi_2.

        :param t: time grid.
        :param psi_1: psi_1 on t.
        :param psi_2: psi_2 on t.
        :param f2: ∫ V_s ds coefficient in the characteristic function.
        :param T: the maturity from the characteristic function.
        :return: the value of the function F on the time grid `t`.
        """
        # we have sigma(T - t) and xi(T - t) in the Riccati equation!
        sigmas = self.sigmas(T - t)
        xi = self.forward_variance(T - t) if self.normalize_variance else 1
        log_normal_drift = psi_1 * (self.model_type == "log-normal")
        return _jit_riccati_func(
            psi_1=psi_1,
            psi_2=psi_2,
            f2=f2,
            sigmas=sigmas,
            xi=xi,
            log_normal_drift=log_normal_drift,
            R=self.R,
            nu=self.nu,
            lam=self.lam,
            rhos_WB=self.rhos_WB
        )

    def __psi_2(
            self,
            t_grid: NDArray[float_],
            psi_1: NDArray[complex_],
            u2: complex,
            f2: complex,
            T: float,
            scheme: str = "exp"
    ) -> NDArray[complex_]:
        """
        Calculates psi_2 as a solution to the vector Riccati equation

        (ψ^i)'(t) = -x_i * ψ^i(t) + F(ψ_1(t), Σ c_j * ψ^j(t)),
        (ψ^i)'(0) = u_2,

        where the function F is self._riccati_func().

        :param t_grid: time grid.
        :param psi_1: psi_1 on t.
        :param u2: V_T coefficient in the characteristic function.
        :param f2: ∫ V_s ds coefficient in the characteristic function.
        :param T: the maturity from the characteristic function.
        :param scheme: numerical scheme for the Riccati equation. Either "exp" or "semi-implicit".
        :return: psi_2 on `t_grid` as an NDArray of size (len(t_grid), len(self.x))
        """
        sigmas_arr = self.sigmas(T - t_grid)
        xi_arr = self.forward_variance(T - t_grid) if self.normalize_variance else np.ones_like(t_grid)
        log_normal_drift_arr = psi_1 * (self.model_type == "log-normal")
        if scheme == "adams":
            return _jit__psi_2_adams(
                t_grid=t_grid,
                psi_1=psi_1,
                u2=u2,
                f2=f2,
                x=self.x,
                c=self.c,
                sigmas_arr=sigmas_arr,
                xi_arr=xi_arr,
                log_normal_drift_arr=log_normal_drift_arr,
                R=self.R,
                nu=self.nu,
                lam=self.lam,
                rhos_WB=self.rhos_WB
            )
        else:
            return _jit__psi_2(
                t_grid=t_grid,
                psi_1=psi_1,
                u2=u2,
                f2=f2,
                scheme=scheme,
                x=self.x,
                c=self.c,
                sigmas_arr=sigmas_arr,
                xi_arr=xi_arr,
                log_normal_drift_arr=log_normal_drift_arr,
                R=self.R,
                nu=self.nu,
                lam=self.lam,
                rhos_WB=self.rhos_WB
            )

    def _char_func(
        self,
        T: float,
        x: float,
        u1: complex,
        u2: complex = 0,
        f1: complex = 0,
        f2: complex = 0,
        cf_timestep: float = 0.001,
        max_grid_size: int = 10**8,
        scheme: str = "exp",
        **kwargs
    ) -> complex:
        """
        Computes the generalized characteristic function

        E[exp{i * u1 * X_T + i * u2 * V_T + i * f1 * ∫ X_s ds + i * f2 * ∫ V_s ds}]     (1)

        for the given model, where X_t = F_t if `model_type` == "normal" and
        X_t = log(F_t) if `model_type` == "log-normal".

        :param u1: X_T coefficient in the characteristic function, see (1).
        :param u2: V_T coefficient in the characteristic function, see (1).
        :param f1: ∫ X_s ds coefficient in the characteristic function, see (1).
        :param f2: ∫ V_s ds coefficient in the characteristic function, see (1).
        :param T: date in the characteristic function, see (1).
        :param scheme: numerical scheme for the Riccati equation. Either "exp", "semi-implicit", or "adams".
        :param x: X_0, equals to F_0 if `model_type` == "normal" and to log(F_0) if `model_type` == "log-normal".
        :param timestep: a timestep to be used in numerical scheme for the Riccati equation.
        :return: a value of the characteristic function (1) for the given coefficients.
        """
        u1, u2, f1, f2 = 1j * u1, 1j * u2, 1j * f1, 1j * f2
        timestep = max(min(cf_timestep, T / 10), T / max_grid_size)

        t_grid = np.linspace(0, T, ceil(T / timestep) + 1)

        psi_1 = _jit__psi_1(t_grid, u1, f1)
        psi_2 = self.__psi_2(t_grid, psi_1, u2, f2, T, scheme=scheme)
        F = self._riccati_func(
            t=t_grid,
            psi_1=psi_1,
            psi_2=psi_2 @ self.c,
            f2=f2,
            T=T
        )
        g0 = self.g0(t_grid)
        f_g0_conv = np.trapz(np.flip(F) * g0, x=t_grid)
        return np.exp(psi_1[-1] * x + u2 * g0[-1] + f_g0_conv)

    def get_corr_mat(self) -> NDArray[float_]:
        """
        :return: a complete correlation matrix of the Brownian motion (W_t, B_t).
        """
        corr_mat = np.eye(self.n_hist_factors + 1)
        corr_mat[:self.n_hist_factors, :self.n_hist_factors] = self.R
        corr_mat[-1, :-1] = self.rhos_WB
        corr_mat[:-1, -1] = self.rhos_WB
        return corr_mat

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
        if B_traj is None:
            # simulation of B_traj
            diffusion = Diffusion(t_grid=t_grid, dim=1, size=size, rng=rng)
            B_traj = diffusion.brownian_motion()[:, 0, :]  # shape (size, len(t_grid))
        else:
            if B_traj.shape != (size, len(t_grid)):
                raise ValueError("Inconsistent dimensions of B_traj were given.")

        # simulation of U
        U_traj = np.zeros((size, self.n_stoch_factors, len(t_grid)))
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

    def get_price_trajectory(
            self,
            t_grid: NDArray[float_],
            size: int,
            F0: Union[float, NDArray[float_]],
            rng: np.random.Generator = None,
            scheme: str = "exp",
            return_variance: bool = False,
            return_factors: bool = False,
            **kwargs
    ) -> Union[NDArray[float_], Tuple[NDArray[float_], ...]]:
        """
        Simulates the underlying price trajectories on the given time grid.

        :param t_grid: time grid.
        :param size: number of trajectories to simulate.
        :param F0: initial value of the underlying price. Without the Kenma-Vorst approximation, F0 is the initial curve of the instantaneous forward (array).
        :param rng: random number generator to simulate the trajectories with.
        :param scheme: a discretization Monte Carlo scheme for the variance, either "exp", or "semi-implicit".
        :param return_variance: whether to return the variance V together with the prices.
        :param return_factors: whether to return the factors U together with the prices.
        :return: an array `F_traj` of shape (size, len(t_grid)) of simulated price trajectories,
            an array `V_traj` of shape (size, len(t_grid)) of variance trajectories if `return_variance` == True,
            an array `U_traj` of shape(size, n_stoch_factors, len(t_grid)) for U if `return_factors` == True.
        """

        if rng is None:
            rng = np.random.default_rng(seed=DEFAULT_SEED)

        diffusion = Diffusion(t_grid=t_grid, dim=self.n_hist_factors + 1, size=size, rng=rng)

        corr_mat = self.get_corr_mat()

        brownian_motion = diffusion.brownian_motion(correlation=corr_mat)
        dW_traj = np.diff(brownian_motion[:, :self.n_hist_factors, :],
                          axis=2)  # shape (size, n_hist_factors, len(t_grid)-1)
        B_traj = brownian_motion[:, self.n_hist_factors, :]  # shape (size, len(t_grid))

        V_traj, U_traj = self.get_variance_trajectory(t_grid=t_grid, size=size, return_factors=True,
                                                      B_traj=B_traj, scheme=scheme)

        xi = self.forward_variance(t_grid[:-1]) if self.normalize_variance else 1
        sigmas = self.sigmas(t_grid[:-1])  # shape (len(t_grid)-1, n_hist_factors)

        if self.model_type == "normal":
            F_traj = F0 * np.ones((size, len(t_grid)))
            F_traj[:, 1:] += np.cumsum(np.einsum('ki,jik->jk', sigmas, dW_traj) * np.sqrt(V_traj[:, :-1] / xi), axis=1)
        elif self.model_type == "log-normal":
            if not self.is_kemna_vorst:
                T_grid = np.array(self.kemna_vorst_grid)
                if T_grid.size != F0.size + 1:
                    raise ValueError('Inconsistent dimensions for the initial curve and the integration grid T_grid.')
                dT = np.diff(T_grid)
                dt = np.diff(t_grid)
                log_F_traj = np.log(F0) * np.ones((size, len(t_grid), F0.size))
                log_F_traj[:, 1:, :] += np.cumsum(np.einsum('mkj,mk->mkj', np.einsum('kij,mik->mkj', sigmas, dW_traj), np.sqrt(V_traj[:, :-1] / xi)), axis=1)\
                    -0.5*np.cumsum(np.einsum('kj, mk -> mkj', np.einsum('kj, k -> kj',
                    np.einsum('kij, kji -> kj', sigmas, sigmas.transpose(0, 2, 1) @ self.R), dt),
                    V_traj[:, :-1] / xi), axis=1)
                
                F_traj = np.zeros((size, len(t_grid)))
                F_traj[:, 0] = np.sum(F0 * dT) / (T_grid[-1] - T_grid[0])
                # return np.exp(log_F_traj)
                F_traj[:, 1:] = np.einsum('mkj,j->mk', np.exp(log_F_traj)[:, 1:, :], dT) / np.sum(dT)
            else:
                log_F_traj = np.log(F0) * np.ones((size, len(t_grid)))
                log_F_traj[:, 1:] += np.cumsum(np.einsum('ki,jik->jk', sigmas, dW_traj) * np.sqrt(V_traj[:, :-1] / xi), axis=1) \
                                        - 0.5 * np.cumsum(np.sum((t_grid[1:] - t_grid[:-1]) * sigmas.T * (self.R @ sigmas.T), axis=0)\
                                                           * V_traj[:, :-1] / xi, axis=1)
                F_traj = np.exp(log_F_traj)
        else:
            raise ValueError("`model_type` should be either 'normal' or 'log-normal'.")

        if return_variance and return_factors:
            return F_traj, V_traj, U_traj
        if return_variance:
            return F_traj, V_traj
        if return_factors:
            return F_traj, U_traj
        return F_traj

    def get_vol_expansion_coefficients(
        self,
        T_arr: NDArray[float_],
        dt: float = 0.001
    ) -> Tuple[Union[NDArray[float_], float], ...]:
        """
        Calculates numerically the Bergomi-Guyon expansion coefficients.

        :param T_arr: array of maturities.
        :param dt: time step used for numerical integration.
        :return: a tuple of coefficients (C_x_xi, C_xi_xi, C_mu, VS vol, vol of vol) corresponding to
            the maturities provided in `T_arr`.
        """
        t_grids = [np.linspace(0, T, ceil(T / dt)) for T in T_arr]
        sigmas_arr = [self.sigmas(t_grid) for t_grid in t_grids]
        alphas = self.__resolvent_alphas / self.lam
        betas = self.__resolvent_betas
        xi_arr = None
        C_x_xi, C_xi_xi, C_mu, v = _jit_semi_analytic_coefficients(
            T_arr=np.array(T_arr),
            sigmas_arr=sigmas_arr,
            t_grids=t_grids,
            rhos=self.rhos_WB,
            R=self.R,
            alphas=alphas,
            betas=betas,
            V0=self.V0,
            theta=self.theta,
            xi_arr=xi_arr
        )
        return C_x_xi, C_xi_xi, C_mu, v, self.nu


@jit(nopython=True)
def _jit__psi_1(
        t_grid: NDArray[float_],
        u1: complex,
        f1: complex
) -> NDArray[complex_]:
    """
    First function in the Riccati equation.

    :param t_grid: time grid.
    :param u1: X_T coefficient in the characteristic function.
    :param f1: ∫ X_s ds coefficient in the characteristic function.
    :return: the value of psi_1 on `t_grid`.
    """
    return u1 + f1 * t_grid


@jit(nopython=True)
def _jit_riccati_func(
        psi_1: NDArray[complex_],
        psi_2: NDArray[complex_],
        f2: complex,
        sigmas: NDArray[float_],
        xi: NDArray[float_],
        log_normal_drift: NDArray[complex_],
        R: NDArray[float_],
        nu: float,
        lam: float,
        rhos_WB: NDArray[float_]
) -> NDArray[complex_]:
    """
    Cf. the function _riccati_func.
    """
    return f2 + 0.5 * np.sum(sigmas.T * (R @ sigmas.T), 0) / xi * (psi_1 ** 2 - log_normal_drift) + \
        psi_2 * (nu * sigmas @ rhos_WB / np.sqrt(xi) * psi_1 - lam) + 0.5 * nu ** 2 * psi_2 ** 2


@jit(nopython=True)
def _jit__psi_2(
        t_grid: NDArray[float_],
        psi_1: NDArray[complex_],
        u2: complex,
        f2: complex,
        scheme: str,
        x: NDArray[float_],
        c: NDArray[float_],
        sigmas_arr: NDArray[float_],
        xi_arr: NDArray[float_],
        log_normal_drift_arr: NDArray[complex_],
        R: NDArray[float_],
        nu: float,
        lam: float,
        rhos_WB: NDArray[float_]
) -> NDArray[complex_]:
    """
    Cf. the function __psi_2.
    """
    psi_2 = np.zeros((t_grid.size, x.size), dtype=np.complex128)
    psi_2[0] = u2
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
            scale_2 = scale_1 * dt
        else:
            scale_2 = (1 - scale_1) / x_row
    else:
        raise ValueError("Incorrect value of `scheme` was given.")
    a_riccati = 0.5 * nu ** 2
    b_riccati = (nu * sigmas_arr @ rhos_WB / np.sqrt(xi_arr) * psi_1 - lam)
    c_riccati = (f2 + 0.5 * np.sum(sigmas_arr.T * (R @ sigmas_arr.T), 0) / xi_arr *
                 (psi_1 ** 2 - log_normal_drift_arr))

    for i in range(len(t_grid) - 1):
        psi_2_i = np.sum(psi_2[i] * c)
        F_i = a_riccati * psi_2_i ** 2 + b_riccati[i] * psi_2_i + c_riccati[i]
        psi_2[i + 1] = scale_1[i] * psi_2[i] + scale_2[i] * F_i
    return psi_2


@jit(nopython=True)
def _jit__psi_2_adams(
        t_grid: NDArray[float_],
        psi_1: NDArray[complex_],
        u2: complex,
        f2: complex,
        x: NDArray[float_],
        c: NDArray[float_],
        sigmas_arr: NDArray[float_],
        xi_arr: NDArray[float_],
        log_normal_drift_arr: NDArray[complex_],
        R: NDArray[float_],
        nu: float,
        lam: float,
        rhos_WB: NDArray[float_]
) -> NDArray[complex_]:
    """
    Adams numerical scheme for psi_2, cf. the function __psi_2.
    """
    psi_2 = np.zeros((t_grid.size, x.size), dtype=np.complex128)
    psi_2[0] = u2
    dt = np.reshape(np.diff(t_grid), (-1, 1))
    x_row = np.reshape(x, (1, -1))

    I1 = (1 - np.exp(- x_row * dt)) / x_row
    I2 = (1 - I1 / dt) / x_row
    gamma = I2 @ c
    exp_scale = np.exp(-x_row * dt)

    a_riccati = 0.5 * nu ** 2
    b_riccati = (nu * sigmas_arr @ rhos_WB / np.sqrt(xi_arr) * psi_1 - lam)
    c_riccati = (f2 + 0.5 * np.sum(sigmas_arr.T * (R @ sigmas_arr.T), 0) / xi_arr *
                 (psi_1 ** 2 - log_normal_drift_arr))
    for i in range(len(t_grid) - 1):
        psi_2_i = np.sum(psi_2[i] * c)
        F_i = a_riccati * psi_2_i**2 + b_riccati[i] * psi_2_i + c_riccati[i]

        c_adams = np.sum(c * exp_scale[i] * psi_2[i]) + np.sum(c * (I1[i] - I2[i])) * F_i

        a_eq = gamma[i] * a_riccati
        b_eq = gamma[i] * b_riccati[i + 1] - 1
        c_eq = c_adams + gamma[i] * c_riccati[i + 1]
        # Solution to the quadratic equation a_eq * psi_2**2 + b_eq * psi_2 + c_eq = 0
        psi_predictor = (-b_eq - np.sqrt(b_eq**2 - 4 * a_eq * c_eq)) / (2 * a_eq)

        # The value of the Riccati function F is implied from the quadratic equation on psi_2
        F_predictor = (psi_predictor - c_adams) / gamma[i]

        psi_2[i + 1] = exp_scale[i] * psi_2[i] + F_i * I1[i] + I2[i] * (F_predictor - F_i)
    return psi_2


@jit(nopython=True)
def _trapz_2d(
    y: NDArray[float_],
    x: NDArray[float_]
):
    """
    jit implementation of np.trapz along axis 0.
    """
    res = np.zeros(y.shape[1])
    for i in range(y.shape[1]):
        res[i] = np.trapz(y[:, i], x)
    return res


@jit(nopython=True)
def _jit_semi_analytic_coefficients(
    T_arr: NDArray[float_],
    sigmas_arr: List[NDArray[float_]],
    t_grids: List[NDArray[float_]],
    rhos: NDArray[float_],
    R: NDArray[float_],
    alphas: NDArray[float_],
    betas: NDArray[float_],
    V0: float,
    theta: float,
    xi_arr: List[NDArray[float_]] = None
):
    v = np.zeros(len(T_arr))
    C_x_xi = np.zeros(len(T_arr))
    C_xi_xi = np.zeros(len(T_arr))
    C_mu = np.zeros(len(T_arr))
    for i in range(len(T_arr)):
        t_grid = t_grids[i]
        if xi_arr is None:
            xi = _jit_forward_variance(
                V0=V0,
                theta=theta,
                alphas=alphas,
                betas=betas,
                t_grid=t_grid
            )
        else:
            xi = xi_arr[i]
        v[i] = _jit_quadratic_variation(sigmas=sigmas_arr[i], R=R, xi=xi, t_grid=t_grid)
        inst_variance = np.sum(sigmas_arr[i].T * (R @ sigmas_arr[i].T), axis=0)
        cov_arr = sigmas_arr[i] @ rhos
        u_grid = np.reshape(t_grid, (1, -1, 1))
        s_grid = np.reshape(t_grid, (1, 1, -1))
        inner_integral = _trapz_2d(
            y=np.sum(
                np.reshape(-alphas, (-1, 1, 1)) * np.exp(np.reshape(betas, (-1, 1, 1)) *
                                                         (u_grid - s_grid)) * (u_grid >= s_grid),
                axis=0
            ) * np.reshape(inst_variance, (-1, 1)),
            x=t_grid
        )
        inner_integral_2 = _trapz_2d(
            y=np.sum(
                np.reshape(-alphas, (-1, 1, 1)) * np.exp(np.reshape(betas, (-1, 1, 1)) *
                                                         (u_grid - s_grid)) * (u_grid >= s_grid),
                axis=0
            ) * np.reshape(inner_integral * cov_arr, (-1, 1)),
            x=t_grid
        )
        C_x_xi[i] = np.trapz(
            y=xi * inner_integral * cov_arr,
            x=t_grid
        )
        C_xi_xi[i] = np.trapz(
            y=xi * inner_integral ** 2,
            x=t_grid
        )
        C_mu[i] = np.trapz(
            y=xi * cov_arr * inner_integral_2,
            x=t_grid
        )
    return C_x_xi, C_xi_xi, C_mu, v


@jit(nopython=True)
def _jit_quadratic_variation(
    sigmas: NDArray[float_],
    R: NDArray[float_],
    xi: NDArray[float_],
    t_grid: NDArray[float_]
):
    """
    Cf. the function `quadratic_variation`.
    """
    inst_variance = np.sum(sigmas.T * (R @ sigmas.T), axis=0)
    return np.trapz(inst_variance * xi, t_grid)


@jit(nopython=True)
def _jit_forward_variance(
    V0: float,
    theta: float,
    alphas: NDArray[float_],
    betas: NDArray[float_],
    t_grid: NDArray[float_]
):
    """
    Cf. the function `forward_variance`.
    """
    return V0 + (V0 - theta) * np.sum(alphas / betas * (np.exp(betas * np.reshape(t_grid, (-1, 1))) - 1),
                                      axis=1)
