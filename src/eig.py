"""EIG estimators and a generic KL-ball worst-case expectation.

Baseline model (linear-Gaussian, 1-D parameter):
    theta ~ N(0, s_theta^2),   y | theta, x ~ N(x * theta, s_noise^2)
Closed form:
    EIG(x) = 0.5 * log(1 + x^2 * s_theta^2 / s_noise^2)
"""
import numpy as np
from scipy.special import logsumexp


def eig_closed_form(x, s_theta=1.0, s_noise=1.0):
    return 0.5 * np.log1p((np.asarray(x) ** 2) * s_theta**2 / s_noise**2)


def _log_lik(y, theta, x, s_noise):
    return -0.5 * np.log(2 * np.pi * s_noise**2) - 0.5 * ((y - x * theta) / s_noise) ** 2


def eig_samples_nmc(x, n_outer=2000, n_inner=2000, s_theta=1.0, s_noise=1.0, rng=None):
    """Per-sample terms of the nested Monte Carlo EIG estimator.

    Returns an array g with g_i = log p(y_i | theta_i) - log p_hat(y_i),
    so that EIG is estimated by g.mean(). Keeping the samples lets us
    reweight them for the robust (KL-ball) version.
    """
    rng = np.random.default_rng(rng)
    theta = rng.normal(0.0, s_theta, n_outer)
    y = x * theta + rng.normal(0.0, s_noise, n_outer)
    theta_in = rng.normal(0.0, s_theta, n_inner)
    log_lik_in = _log_lik(y[:, None], theta_in[None, :], x, s_noise)
    log_evidence = logsumexp(log_lik_in, axis=1) - np.log(n_inner)
    return _log_lik(y, theta, x, s_noise) - log_evidence


def kl_worst_case_mean(g, eps):
    """inf { E_q[g] : KL(q || p_hat) <= eps }, p_hat = empirical dist. of g.

    Uses the standard dual:
        inf_q E_q[g] = sup_{lam > 0}  -lam * log E_p[exp(-g / lam)] - lam * eps
    NOTE: check against the exact definition of robust EIG in Go & Isaac (2022)
    before relying on this; it is the generic KL-ball lower bound.
    Returns (value, lam_star).
    """
    from scipy.optimize import minimize_scalar

    g = np.asarray(g, dtype=float)
    if eps <= 0:
        return g.mean(), np.inf
    n = g.size

    def neg_dual(log_lam):
        lam = np.exp(log_lam)
        return -(-lam * (logsumexp(-g / lam) - np.log(n)) - lam * eps)

    res = minimize_scalar(neg_dual, bounds=(-10, 10), method="bounded")
    val = -res.fun
    return min(val, g.mean()), float(np.exp(res.x))
