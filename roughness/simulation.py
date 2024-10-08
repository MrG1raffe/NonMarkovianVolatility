import numpy as np
from fbm import FBM


def simulate_BM(n_sample, T=1):
    """
    Samples standard Brownian motion on [0, T].

    Args:
        n_samples: number of points in the trajectory.
        T: time segment size.

    Returns:
        Trajectory of the Brownian motion sampled at (i / (n_sample - 1)) * T, i = 0, ..., n_sample.
    """
    step = T / (n_sample - 1)
    W = np.concatenate([[0], np.sqrt(step) * np.random.randn(n_sample - 1)]).cumsum()
    return W


def simulate_gBM(n_sample, sigma=1, init=1, T=1, is_mart=True):
    """
    Samples a martingale geometric Brownian motion on [0, T], satisfying SDE
    dX = sigma*X*dW, X_0 = init.

    Args:
        n_samples: number of points in the trajectory.
        sigma: volatility parameter (see SDE).
        init: value of the process at t = 0.
        is_mart: whether the process should be a martingale or exp(sigma*W).
        T: segment size.

    Returns:
        Trajectory of the geometric Brownian motion sampled at (i / (n_sample - 1)) * T, i = 0, ..., n_sample.
    """
    W = simulate_BM(n_sample, T=T)
    t = np.linspace(0, T, n_sample)
    gBM = init*np.exp(-0.5 * sigma**2 * t * is_mart + sigma * W)
    return gBM


def simulate_OU(n_sample, sigma=1, lam=1, mu=0, T=1, init=None):
    """
    Samples Ornstein–Uhlenbeck process on [0, T], satisfying SDE
    dX = -lam*(X - mu)*dt + sigma*dW, X_0 = init.

    Args:
        n_samples: number of points in the trajectory.
        sigma: volatility parameter (see SDE).
        lam: mean-reversion speed parameter (see SDE).
        mu: process mean (see SDE).
        T: time segmet size.
        init: value of the process at t = 0. If None, X_0 is sampled from
            N(0, sigma**2 / (2 * lam)) for the process to be stationary.

    Returns:
        Trajectory of the Ornstein–Uhlenbeck process sampled at (i / (n_sample - 1)) * T, i = 0, ..., n_sample.
    """
    sigma = sigma / np.sqrt(2 * lam)  # the standard deviation of X_t
    k = int(np.ceil(np.log2(n_sample)))
    T = (2**k / n_sample) * T
    ou = np.zeros(2**k + 1)
    ou[0] = np.random.randn() * sigma if init is None else 0
    ou[-1] = sigma * np.sqrt(1 - np.exp(-2 * lam * T)) * np.random.randn() + ou[0] * np.exp(-lam * T)
    step = 2**k
    first = 2**(k - 1)
    h = T
    for i in range(0, k):
        h /= 2
        ou[first::step] = np.random.randn(2**i) * sigma * \
            np.sqrt((1 - np.exp(-2 * h * lam)) / (1 + np.exp(-2 * h * lam))) + \
            (ou[:-1:step] + ou[step::step]) * np.exp(-lam * h) / (1 + np.exp(-2 * lam * h))
        first = first // 2
        step = step // 2
    return (mu + ou)[:n_sample]


def simulate_fOU(n_sample, H=0.5, sigma=1, lam=1, mu=0, T=1, init=0):
    """
    Samples fractional Ornstein–Uhlenbeck process on [0, T], satisfying SDE
    dX = -lam*(X - mu)*dt + sigma*dW^H, X_0 = init.

    Args:
        n_samples: number of points in the trajectory.
        H: hurst parameter.
        sigma: volatility parameter (see SDE).
        lam: mean-reversion speed parameter (see SDE).
        mu: process mean (see SDE).
        T: time segmet size.
        init: value of the process at t = 0.

    Returns:
        Trajectory of the fractional Ornstein–Uhlenbeck process sampled at (i / (n_sample - 1)) * T, i = 0, ..., n_sample.
    """
    fGN = np.diff(FBM(n_sample-1, H, T).fbm()) * sigma
    delta = T / (n_sample - 1)
    # ToDo: rewrite as a loop. It works faster!
    ufun = np.frompyfunc(lambda x, y: (1 - lam*delta)*x + y, 2, 1)
    b = np.concatenate([[init], fGN])
    return np.array(ufun.accumulate(b, dtype=np.object) + mu, dtype='float')


def simulate_OUOU(n_sample, sigma_x=1, sigma_y=1, lam=1, beta=1, mu=0, T=1, init_y=None, init=0):
    """
    Samples double Ornstein–Uhlenbeck process on [0, T], satisfying SDE
    dX = -lam*(X - Y)*dt + sigma_x*dW, X_0 = init,
    dY = -beta*Y*dt + sigma_y*dB.
    Standard Euler's method is used for sampling.

    Args:
        n_samples: number of points in the trajectory.
        sigma_x: volatility parameter of X (see SDE).
        sigma_y: volatility parameter of Y (see SDE).
        lam: mean-reversion speed parameter of X (see SDE).
        beta: mean-reversion speed parameter of Y (see SDE).
        T: time segment size.
        init_y: initial value of the Y. If None, Y_0 is sampled from
            N(0, sigma_y**2 / (2 * beta)) for the process Y to be stationary.
        init: value of the process X at t = 0.

    Returns:
        Trajectory of the double Ornstein–Uhlenbeck process and process Y sampled at (i / (n_sample - 1)) * T, i = 0, ..., n_sample.
    """
    Y = simulate_OU(n_sample-1, sigma=sigma_y, lam=beta, mu=mu, T=T, init=init_y)
    delta = T / (n_sample - 1)
    # ToDo: rewrite as a loop. It works faster!
    b = np.concatenate([[init], sigma_x * np.sqrt(delta) * np.random.randn(Y.size) + lam * delta * Y])
    ufun = np.frompyfunc(lambda x, y: (1 - lam*delta)*x + y, 2, 1)
    return ufun.accumulate(b, dtype=np.object).astype(float), Y


def simulate_OUOU_kernel(n_sample, sigma_x, sigma_y, lam_x, lam_y, mu=0, x0=0, y0=None, T=1):
    """
    Samples double Ornstein–Uhlenbeck process on [0, T], satisfying SDE
    dX = -lam_x*(X - Y)*dt + sigma_x*dW, X_0 = x0,
    dY = -lam_y*(Y - mu)*dt + sigma_y*dB, Y_0 = y0.
    Transition density is used for sampling.

    Args:
        n_samples: number of points in the trajectory.
        sigma_x: volatility parameter of X (see SDE).
        sigma_y: volatility parameter of Y (see SDE).
        lam_x: mean-reversion speed parameter of X (see SDE).
        lam_y: mean-reversion speed parameter of Y (see SDE).
        mu: process mean (see SDE).
        T: time segment size.
        x0: value of the process X at t = 0.
        y0: initial value of the Y. If None, Y_0 is sampled from
            N(mu, sigma_y**2 / (2 * lam_y)) for the process Y to be stationary.

    Returns:
        Trajectory of the double Ornstein–Uhlenbeck process and process Y.
    """
    def transition_mean(t, x0, y0):
        m_x = np.exp(-lam_x*t)*x0 + (lam_x*y0) / (lam_x - lam_y)*(np.exp(-lam_y*t) - np.exp(-lam_x*t))
        m_y = np.exp(-lam_y*t)*y0
        return [m_x, m_y]

    def transition_covmat(t, x0, y0):
        m_x, m_y = transition_mean(t, x0, y0)
        s = sigma_y**2 / (2*lam_y)
        q_y = s + (y0**2 - s)*np.exp(-2*lam_y*t)
        c = (x0*y0 + lam_x*(s / (lam_x + lam_y) * (np.exp((lam_x + lam_y)*t) - 1) + (y0**2 - s) /
                            (lam_x - lam_y)*(np.exp((lam_x - lam_y)*t) - 1))) * np.exp(-(lam_x + lam_y)*t)
        I_val = 2*lam_x / (lam_y - lam_x) * (x0*y0 - np.exp(2*lam_x*t)*c + 0.5*s*(np.exp(2*lam_x*t) - 1) + 0.5*(y0**2 - s) *
                                             lam_x / (lam_x - lam_y) * (np.exp(2*(lam_x - lam_y)*t) - 1))
        q_x = np.exp(-2*lam_x*t) * (x0**2 + sigma_x**2 / (2*lam_x) * (np.exp(2*lam_x*t) - 1) + I_val)
        covmat = np.array([
            [q_x - m_x**2, c - m_x*m_y],
            [c - m_x*m_y, q_y - m_y**2]
        ])
        return covmat

    if y0 is None:
        y0 = mu + np.random.randn() * sigma_y / np.sqrt(2*lam_y)

    x = np.zeros(n_sample)
    y = np.zeros_like(x)
    x[0] = x0 - mu
    y[0] = y0 - mu

    dt = T / (n_sample - 1)
    x_cur, y_cur = x[0], y[0]
    for i in range(1, n_sample):
        x_cur, y_cur = np.random.multivariate_normal(transition_mean(dt, x_cur, y_cur),
                                                     transition_covmat(dt, x_cur, y_cur))
        x[i] = x_cur
        y[i] = y_cur

    return x + mu, y + mu
